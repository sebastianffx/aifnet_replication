#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import nibabel as nib
from scipy import ndimage
import glob
import sys,os
from natsort import natsorted
import tensorflow_probability as tfp
import random
from aifnet_utils.preprocess import read_nifti_file, normalize, normalize_aif, process_scan, normalize_zero_one
from aifnet_utils.losses import MaxCorrelation
from aifnet_utils.data_loaders import read_isles_volumepaths_from_file_otf, read_isles_annotations_from_file, ISLES18DataGen_aifvof_otf
from aifnet_utils.data_loaders import delay_sequence_padding, anticipate_sequence_padding, late_bolus, early_bolus
from aifnet_utils.results import plot_predictions
from aifnet_utils.models_aifnet import get_model_twoPvols
import gc


#get_ipython().run_line_magic('matplotlib', 'inline')

#get_ipython().system('pwd')





keras.backend.set_image_data_format('channels_last')
ROOT_EXP = '/home/sebastian/experiments/aifnet_replication/'
root_dir     = '/media/sebastian/data/ASAP/ISLES2018_Training'
#At insel: /media/sebastian/data/ASAP/ISLES2018_Training
#Local: '/Users/sebastianotalora/work/postdoc/data/ISLES/'
aif_annotations_path = ROOT_EXP + 'annotated_aif_vof_complete_revised.csv'#'radiologist_annotations.csv'#'annotated_aif_vof_complete_revised.csv'
min_num_volumes_ctp = 28
#ROOT_EXP = '/home/sebastian/experiments/aifnet_replication'


nb_epochs=8
lrs = [0.01, 0.1, 0.00001, 0.0001, 0.001]
random_lrs = [random.uniform(0,lr)*2 for lr in lrs]

for lr in lrs:
    for current_fold in range(1,6):        
        print('======= TRAINING FOR THE FOLD ' + str(current_fold) + ' (LR)  '+ str(lr) +'=======')
        prediction_meassures = []
        prediction_ids = []

        #Reading AIFs and VOFs for each of the partitions
        train_partition_path = ROOT_EXP+'/partitions/fold_'+str(current_fold) +'/train_v2.txt'
        valid_partition_path = ROOT_EXP+'/partitions/fold_'+str(current_fold) +'/valid_v2.txt'
        test_partition_path =  ROOT_EXP+'/partitions/fold_'+str(current_fold) +'/test_v2.txt'

        aif_annotations_train, vof_annotations_train = read_isles_annotations_from_file(aif_annotations_path, train_partition_path, 
                                                        root_dir, min_num_volumes_ctp, return_aif_only = False)
        aif_annotations_valid, vof_annotations_valid = read_isles_annotations_from_file(aif_annotations_path, valid_partition_path, root_dir, 
                                                min_num_volumes_ctp, return_aif_only = False)
        aif_annotations_test, vof_annotations_test = read_isles_annotations_from_file(aif_annotations_path,  test_partition_path,
                                                root_dir, min_num_volumes_ctp, return_aif_only = False)



        ctp_volumes_train = read_isles_volumepaths_from_file_otf(root_dir, train_partition_path, aif_annotations_path)
        ctp_volumes_valid = read_isles_volumepaths_from_file_otf(root_dir, valid_partition_path, aif_annotations_path)
        ctp_volumes_test = read_isles_volumepaths_from_file_otf(root_dir, test_partition_path, aif_annotations_path)

        print(len(ctp_volumes_train), len(aif_annotations_train))
        print(len(ctp_volumes_valid), len(aif_annotations_valid))
        print(len(ctp_volumes_test), len(aif_annotations_test))


        # Build model.
        model = get_model_twoPvols(width=256, height=256, num_channels=min_num_volumes_ctp)
        model.summary()
        #tf.keras.utils.plot_model(
        #    model,
        #    to_file="aifnet.png",
        #    show_shapes=False,
        #    show_dtype=False,
        #    show_layer_names=True,
        #    rankdir="TB",
        #    expand_nested=False,
        #    dpi=96)

        initial_learning_rate = lr#0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100, decay_rate=0.96, staircase=True
        )


        optimizer_aifnet = optimizer = keras.optimizers.SGD(learning_rate=lr_schedule) #keras.optimizers.Adam(learning_rate=initial_learning_rate)
        model.compile(
            loss=['mse','mse'],
            optimizer=optimizer_aifnet,
            metrics=['mae'])


        # Define callbacks.
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_aif_loss_mae", patience=3)
        path_checkpointer_model = ROOT_EXP +'/results/' 
        path_checkpointer_model += 'aifnet_2Pvols_SGD_mse_augment_lr' + str(initial_learning_rate) + '_fold_' + str(current_fold) +'.hdf5'
        path_tensorboard_log    = ROOT_EXP + '/results/logsTensorBoard/'
        path_tensorboard_log    += 'aifnet_2Pvols_SGD_mse_augment_lr' + str(initial_learning_rate) + '_fold_' + str(current_fold) 

        checkpointer = ModelCheckpoint(filepath=path_checkpointer_model, monitor='val_aif_loss_mae', 
                                    verbose=1, save_best_only=True)
        tb_callback = TensorBoard(log_dir=path_tensorboard_log, histogram_freq=0, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)



        train_datagen = ISLES18DataGen_aifvof_otf(ctp_volumes=ctp_volumes_train, annotations_aif=aif_annotations_train,
                                    annotations_vof=vof_annotations_train,minimum_number_volumes_ctp = min_num_volumes_ctp, batch_size=1,
                                                time_arrival_augmentation = True)
        validation_datagen =  ISLES18DataGen_aifvof_otf(ctp_volumes=ctp_volumes_valid, annotations_aif=aif_annotations_valid,
                                    annotations_vof=vof_annotations_valid,minimum_number_volumes_ctp = min_num_volumes_ctp, batch_size=1,
                                                time_arrival_augmentation = True)

        model.fit(train_datagen,batch_size=1,callbacks=[checkpointer,tb_callback,early_stopping_cb],
                epochs=nb_epochs, validation_data=validation_datagen)


        # print('======= PREDICTING IN THE TEST PARTITION FOR THE FOLD ' + str(current_fold) + ' =======')
        # type_predictions = 'AIF'
        # results_meassures = []
        # for case_number in range(len(ctp_volumes_test)):
        #     case_id = ctp_volumes_test[case_number]['image'].split('.')[-2]
        #     prediction_ids.append(case_id)
        #     cur_nib = nib.load(ctp_volumes_test[case_number]['image'])
        #     ctp_vals = cur_nib.get_fdata()
        #     x = normalize(ctp_vals[:,:,:,0:min_num_volumes_ctp])
        #     if type_predictions == 'AIF':
        #         y = aif_annotations_test[case_id]
        #     if type_predictions == 'VOF':
        #         y = vof_annotations_test[case_id]
        #     prefix_fig = ROOT_EXP + '/results/predictions_aif/'+path_tensorboard_log.split('/')[-1]+'_case_'+str(case_id)
        #     results_meassures.append(plot_predictions(model,x,y, prefix_fig, True, type_predictions,True))

        # preds_fold = tfp.stats.correlation(np.array(results_meassures)[:,1,:],np.array(results_meassures)[:,0,:], sample_axis=0, event_axis=None)
        # preds_fold = preds_fold.numpy()
        # prediction_meassures.append([preds_fold.mean(),preds_fold.std(),preds_fold.var()])

        # np.savetxt('results/pearson_fold_'+str(current_fold)+'.csv', prediction_meassures, delimiter=',',fmt='%1.5f')
        # np.savetxt('results/allpreds_fold_'+str(current_fold)+'.csv', np.array(results_meassures)[:,1,:], delimiter=',',fmt='%1.5f')


        # test_ids_file=open('results/pred_ids_fold_'+str(current_fold)+'.csv','w')
        # for element in prediction_ids:        
        #     test_ids_file.write(element+'\n')
        # test_ids_file.close()
        del ctp_volumes_train
        del ctp_volumes_valid
        del ctp_volumes_test  
        del train_datagen  
        del aif_annotations_train
        del aif_annotations_valid
        del aif_annotations_test
        del model
        keras.backend.clear_session()
        gc.collect()
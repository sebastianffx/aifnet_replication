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
from aifnet_utils.data_loaders import ISLES18DataGen_aif, read_isles_volumepaths_from_file_otf, read_isles_annotations_from_file, ISLES18DataGen_aifvof_otf
from aifnet_utils.data_loaders import delay_sequence_padding, anticipate_sequence_padding, late_bolus, early_bolus
from aifnet_utils.results import plot_predictions
from aifnet_utils.models_aifnet import get_model_onehead 
import gc


#Reading an example PCT volume
LOCATION = 'SERVER'
if LOCATION == 'LOCAL':
    ROOT_EXP = '/Users/sebastianotalora/work/postdoc/ctp/aifnet_replication/'
    root_dir  = '/Users/sebastianotalora/work/postdoc/data/ISLES/'

if LOCATION == 'INSEL':
    ROOT_EXP = '/home/sebastian/experiments/aifnet_replication/'
    root_dir  = '/media/sebastian/data/ASAP/ISLES2018_Training'

IF LOCATION == 'SERVER':
    ROOT_EXP = '/home/sotalora/aifnet_replication/'
    root_dir     = '/data/images/sotalora/ISLES18/'

aif_annotations_path = ROOT_EXP + 'radiologist_annotations.csv'


aif_annotations_path = ROOT_EXP + 'radiologist_annotations_cleaned.csv'#'radiologist_annotations.csv'#'annotated_aif_vof_complete_revised.csv'
min_num_volumes_ctp = 43

nb_epochs=10
lrs = [0.01, 0.1, 0.00001, 0.0001, 0.001]
random_lrs_1 = [random.uniform(0,lr)*2 for lr in lrs]
random_lrs_2 = [random.uniform(0,1) for i in range(10)]

for lr in random_lrs_1 + random_lrs_2:
    for current_fold in range(1,6):        
        print('======= TRAINING FOR THE FOLD ' + str(current_fold) + ' (LR)  '+ str(lr) +'=======')
        prediction_meassures = []
        prediction_ids = []

        #Reading AIFs and VOFs for each of the partitions
        train_partition_path = ROOT_EXP+'/partitions_cleaned/fold_'+str(current_fold) +'/train_v2.txt'
        valid_partition_path = ROOT_EXP+'/partitions_cleaned/fold_'+str(current_fold) +'/valid_v2.txt'
        test_partition_path =  ROOT_EXP+'/partitions_cleaned/fold_'+str(current_fold) +'/test_v2.txt'

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
        model = get_model_onehead(width=256, height=256, num_channels=min_num_volumes_ctp)
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
            loss=['mse'],
            optimizer=optimizer_aifnet,
            metrics=['mae'])


        # Define callbacks.
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_mae", patience=3)
        path_checkpointer_model = ROOT_EXP +'/results/' 
        path_checkpointer_model += 'aifnet_SGD_MSE_augment_lr' + str(initial_learning_rate) + '_fold_' + str(current_fold) +'.hdf5'
        path_tensorboard_log    = ROOT_EXP + '/results/logsTensorBoard/'
        path_tensorboard_log    += 'aifnet_SGD_MSE_augment_lr' + str(initial_learning_rate) + '_fold_' + str(current_fold)

        checkpointer = ModelCheckpoint(filepath=path_checkpointer_model, monitor='val_mae', 
                                    verbose=1, save_best_only=True)
        tb_callback = TensorBoard(log_dir=path_tensorboard_log, histogram_freq=0, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)



        train_datagen = ISLES18DataGen_aif(ctp_volumes=ctp_volumes_train, annotations_aif=aif_annotations_train,
                                    minimum_number_volumes_ctp = min_num_volumes_ctp, batch_size=1,
                                                time_arrival_augmentation = True)
        validation_datagen =  ISLES18DataGen_aif(ctp_volumes=ctp_volumes_valid, annotations_aif=aif_annotations_valid,
                                    minimum_number_volumes_ctp = min_num_volumes_ctp, batch_size=1,
                                                time_arrival_augmentation = True)

        model.fit(train_datagen,batch_size=1,callbacks=[checkpointer,tb_callback,early_stopping_cb],
                epochs=nb_epochs, validation_data=validation_datagen)
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
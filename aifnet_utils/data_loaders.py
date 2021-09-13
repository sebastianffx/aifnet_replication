import nibabel as nib
import glob
import random
import numpy as np
import os
from natsort import natsorted
import tensorflow as tf
from .preprocess import read_nifti_file, normalize, normalize_aif, resize_volume, process_scan, normalize_zero_one

def read_isles_annotations(aif_annotations_path, root_dir, minimum_number_volumes_ctp, return_aif_only = True):
    aif_annotations = []
    aif_annotations_file = open(aif_annotations_path,'r')
    aif_annotations_file.readline()
    cases_annotations = {}
    for line in aif_annotations_file: #Here we substract one to account for 0-indexing in python
        cases_annotations[line.split(',')[0]] = [np.array([int(line.split(',')[1]),int(line.split(',')[2]),int(line.split(',')[3])])-1,
                                                 np.array([int(line.split(',')[4]),int(line.split(',')[5]),int(line.split(',')[6])])-1]
    aif_annotations_file.close()
    
    dataset_dir = os.path.join(root_dir, "TRAINING")
    filenames_4D = natsorted(glob.glob(dataset_dir + "/case_*/*4D*/*nii*"))
    cases_paths = {path.split('.')[-2]: path for path in filenames_4D}

    aifs_cases = {}
    vofs_cases = {}
    for cur_case in cases_annotations.keys():
        #print(cur_case)
        AIFx,AIFy,AIFz = cases_annotations[cur_case][0][0],cases_annotations[cur_case][0][1], cases_annotations[cur_case][0][2]
        VOFx,VOFy,VOFz = cases_annotations[cur_case][1][0],cases_annotations[cur_case][1][1], cases_annotations[cur_case][1][2]
        fname = cases_paths[cur_case]
        cur_nib = nib.load(fname)    
        ctp_vals = cur_nib.get_fdata()
        #Reading the intensity values of the AIF and VOF
        AIF = ctp_vals[AIFx,AIFy,AIFz,:]
        VOF = ctp_vals[VOFx,VOFy,VOFz,:]
        aifs_cases[cur_case] = AIF[0:minimum_number_volumes_ctp] #Since not all the CTP sequences have the same #volumes
        vofs_cases[cur_case] = VOF[0:minimum_number_volumes_ctp] #Since not all the CTP sequences have the same #volumes
    if return_aif_only: 
        return aifs_cases
    else:
        return aifs_cases,vofs_cases

def read_isles_annotations_from_file(aif_annotations_path, partition_file_path, root_dir, minimum_number_volumes_ctp, return_aif_only = True):    
    aif_annotations = []
    aif_annotations_file = open(aif_annotations_path,'r')
    aif_annotations_file.readline()
    cases_annotations = {}

    for line in aif_annotations_file: #Here we substract one to account for 0-indexing in python
        cases_annotations[line.split(',')[0]] = [np.array([int(line.split(',')[1]),int(line.split(',')[2]),int(line.split(',')[3])]),
                                                 np.array([int(line.split(',')[4]),int(line.split(',')[5]),int(line.split(',')[6])])]
    #print(cases_annotations)
    aif_annotations_file.close()
    dataset_dir_test = os.path.join(root_dir, "TESTING")
    dataset_dir = os.path.join(root_dir, "TRAINING")
    filenames_4D = natsorted(glob.glob(dataset_dir + "/case_*/*4D*/*nii*") + glob.glob(dataset_dir_test + "/case_*/*4D*/*nii*"))

    #print(len(filenames_4D))
    cases_paths = {path.split('.')[-2]: path for path in filenames_4D}
    #Reading only the relevant cases from the partition file 
    partition_file = open(partition_file_path,'r')
    relevant_cases = []
    for line in partition_file:
        relevant_cases.append(line.split('.')[-2])
    partition_file.close()
    #print(relevant_cases)
    #print(len(cases_annotations))
    #print(len(relevant_cases))
    #print(len(cases_paths))
    aifs_cases = {}
    vofs_cases = {}
    for cur_case in cases_annotations.keys():
        if cur_case in set(relevant_cases):            
            #print(cur_case + " is relevant")
            AIFx,AIFy,AIFz = cases_annotations[cur_case][0][0],cases_annotations[cur_case][0][1], cases_annotations[cur_case][0][2]
            VOFx,VOFy,VOFz = cases_annotations[cur_case][1][0],cases_annotations[cur_case][1][1], cases_annotations[cur_case][1][2]
            fname = cases_paths[cur_case]
            cur_nib = nib.load(fname)    
            ctp_vals = cur_nib.get_fdata()
            #Reading the intensity values of the AIF and VOF
            AIF = ctp_vals[AIFx,AIFy,AIFz,:]
            VOF = ctp_vals[VOFx,VOFy,VOFz,:]
            aifs_cases[cur_case] = AIF[0:minimum_number_volumes_ctp] #Since not all the CTP sequences have the same #volumes
            vofs_cases[cur_case] = VOF[0:minimum_number_volumes_ctp] #Since not all the CTP sequences have the same #volumes
    if return_aif_only: 
        return aifs_cases
    else:
        return aifs_cases,vofs_cases



def read_isles_volumes(root_dir, aif_annotations_path, min_num_volumes_ctp, take_two_slices_only=False):
    dataset_dir = os.path.join(root_dir, "TRAINING")
    filenames_4D = natsorted(glob.glob(dataset_dir + "/case_*/*4D*/*nii*"))
    cases_paths = {}
    cases_paths = {path.split('.')[-2]: path for path in filenames_4D}
    #This is a little bit awful, but we need to get the coordinates from the annotations file to 
    #know which two slices get exactly
    cases_annotations = {}
    aif_annotations_file = open(aif_annotations_path,'r')
    aif_annotations_file.readline()
    for line in aif_annotations_file: #Here we substract one to account for 0-indexing in python
        cases_annotations[line.split(',')[0]] = [np.array([int(line.split(',')[1]),int(line.split(',')[2]),int(line.split(',')[3])])-1,
                                             np.array([int(line.split(',')[4]),int(line.split(',')[5]),int(line.split(',')[6])])-1]
    
    datalist = []
    
    for cur_case in cases_annotations.keys():
        fname = cases_paths[cur_case]
        cur_nib = nib.load(fname)    
        ctp_vals = cur_nib.get_fdata()
        AIFx,AIFy,AIFz = cases_annotations[cur_case][0][0],cases_annotations[cur_case][0][1], cases_annotations[cur_case][0][2]
        VOFx,VOFy,VOFz = cases_annotations[cur_case][1][0],cases_annotations[cur_case][1][1], cases_annotations[cur_case][1][2]
        #Four cases either is it possible to have the slice up, or the one down, or any of them
        if take_two_slices_only: 
            if ctp_vals.shape[2] != 2 and AIFz+1 <=  ctp_vals.shape[2] and AIFz>0:            
                    ctp_vals = ctp_vals[:,:,AIFz-1:AIFz+1,:]
                    AIFz = 1
                    print("After " + str(ctp_vals.shape))
            if ctp_vals.shape[2] != 2 and AIFz+1 <  ctp_vals.shape[2] and AIFz==0:
                    AIFx,AIFy,AIFz = cases_annotations[cur_case][0][0],cases_annotations[cur_case][0][1], cases_annotations[cur_case][0][2]
                    #print("Adding it in the other direction " + str(ctp_vals.shape))
                    ctp_vals = ctp_vals[:,:,AIFz:AIFz+2,:]
                    AIFz = 0
                    print("After " + str(ctp_vals.shape))
            else:
                ("Not processed")
        datalist.append({"image": fname, "ctpvals": ctp_vals[:,:,:,0:min_num_volumes_ctp]})
    return datalist

def read_isles_volumes_from_file(root_dir,partition_file_path, aif_annotations_path, min_num_volumes_ctp, take_two_slices_only=False):
    dataset_dir = os.path.join(root_dir, "TRAINING")
    dataset_dir_test = os.path.join(root_dir, "TESTING")
    filenames_4D = natsorted(glob.glob(dataset_dir + "/case_*/*4D*/*nii*") + glob.glob(dataset_dir_test + "/case_*/*4D*/*nii*"))
    #print("La longitud de todos los niis es " +str(len(filenames_4D)))
    cases_paths = {}
    cases_paths = {path.split('.')[-2]: path for path in filenames_4D}
    #This is a little bit awful, but we need to get the coordinates from the annotations file to 
    #know which two slices get exactly
    cases_annotations = {}
    aif_annotations_file = open(aif_annotations_path,'r')
    aif_annotations_file.readline()
    for line in aif_annotations_file: #Here we substract one to account for 0-indexing in python
        cases_annotations[line.split(',')[0]] = [np.array([int(line.split(',')[1]),int(line.split(',')[2]),int(line.split(',')[3])])-1,
                                             np.array([int(line.split(',')[4]),int(line.split(',')[5]),int(line.split(',')[6])])-1]
    partition_file = open(partition_file_path,'r')
    relevant_cases = []
    for line in partition_file:
        relevant_cases.append(line.split('.')[-2])
    partition_file.close()

    datalist = []
    
    for cur_case in cases_annotations.keys():
        if cur_case in set(relevant_cases):            
            fname = cases_paths[cur_case]
            cur_nib = nib.load(fname)
            ctp_vals = cur_nib.get_fdata()
            AIFx,AIFy,AIFz = cases_annotations[cur_case][0][0],cases_annotations[cur_case][0][1], cases_annotations[cur_case][0][2]
            VOFx,VOFy,VOFz = cases_annotations[cur_case][1][0],cases_annotations[cur_case][1][1], cases_annotations[cur_case][1][2]
            #Four cases either is it possible to have the slice up, or the one down, or any of them
            if take_two_slices_only: 
                if ctp_vals.shape[2] != 2 and AIFz+1 <=  ctp_vals.shape[2] and AIFz>0:            
                        ctp_vals = ctp_vals[:,:,AIFz-1:AIFz+1,:]
                        AIFz = 1
                        print("After " + str(ctp_vals.shape))
                if ctp_vals.shape[2] != 2 and AIFz+1 <  ctp_vals.shape[2] and AIFz==0:
                        AIFx,AIFy,AIFz = cases_annotations[cur_case][0][0],cases_annotations[cur_case][0][1], cases_annotations[cur_case][0][2]
                        #print("Adding it in the other direction " + str(ctp_vals.shape))
                        ctp_vals = ctp_vals[:,:,AIFz:AIFz+2,:]
                        AIFz = 0
                        print("After " + str(ctp_vals.shape))
                else:
                    ("Not processed")
            datalist.append({"image": fname, "ctpvals": ctp_vals[:,:,:,0:min_num_volumes_ctp]})
    return datalist

def read_isles_volumepaths_from_file_otf(root_dir,partition_file_path, aif_annotations_path):
    dataset_dir = os.path.join(root_dir, "TRAINING")
    dataset_dir_test = os.path.join(root_dir, "TESTING")
    filenames_4D = natsorted(glob.glob(dataset_dir + "/case_*/*4D*/*nii*") + glob.glob(dataset_dir_test + "/case_*/*4D*/*nii*"))
    #print("La longitud de todos los niis es " +str(len(filenames_4D)))
    cases_paths = {}
    cases_paths = {path.split('.')[-2]: path for path in filenames_4D}

    cases_annotations = {}
    aif_annotations_file = open(aif_annotations_path,'r')
    aif_annotations_file.readline()
    for line in aif_annotations_file: #Here we substract one to account for 0-indexing in python
        cases_annotations[line.split(',')[0]] = [np.array([int(line.split(',')[1]),int(line.split(',')[2]),int(line.split(',')[3])])-1,
                                             np.array([int(line.split(',')[4]),int(line.split(',')[5]),int(line.split(',')[6])])-1]
    partition_file = open(partition_file_path,'r')
    relevant_cases = []
    for line in partition_file:
        relevant_cases.append(line.split('.')[-2])
    partition_file.close()

    datalist = []
    
    for cur_case in cases_annotations.keys():
        if cur_case in set(relevant_cases):            
            fname = cases_paths[cur_case]
            datalist.append({"image": fname})
    return datalist


##CPT Augmentation techniques for including in the data loaders
def delay_sequence_padding(sequence,delay_t):
    nb_timepoints = np.array(sequence).shape[0] #This assumes a SINGLE vector with all the timepoints
    delayed_sequence = np.zeros(np.array(sequence).shape)
    delayed_sequence[0:delay_t] = sequence[0] #Repeating the first time point
    delayed_sequence[delay_t:]  = sequence[1:nb_timepoints-delay_t+1]
    return delayed_sequence

def anticipate_sequence_padding(sequence,delay_t):
    nb_timepoints = np.array(sequence).shape[0] #This assumes a SINGLE vector with all the timepoints
    early_intensity = np.zeros(sequence.shape)
    early_intensity[0:nb_timepoints-delay_t] = sequence[delay_t:nb_timepoints] #Shifting the first time points
    early_intensity[nb_timepoints-delay_t:] = sequence[-1]
    return early_intensity

def late_bolus(volume_sequence, labels, delay_t=None):
    labels_shape = np.array(labels).shape
        
    delayed_volume, delayed_intensity = np.zeros(volume_sequence.shape), np.zeros(np.array(labels).shape)
    nb_timepoints = volume_sequence.shape[-1]

    first_volume = volume_sequence[:,:,:,0]    
    if delay_t == None:
        delay_t = random.randint(1,int(nb_timepoints/3))
    if delay_t == 0:
        return volume_sequence, labels
    
    for i in range(0,delay_t+1):
        delayed_volume[:,:,:,i] = first_volume
    #print(i)
    delayed_volume[:,:,:,i:] = volume_sequence[:,:,:,1:nb_timepoints-i+1]
    
    #Delaying each of the labels
    if len(labels_shape)==2:#We are processing the AIF and the VOF
        delayed_aif = delay_sequence_padding(labels[0],delay_t)
        delayed_vof = delay_sequence_padding(labels[1],delay_t)
        delayed_intensity = np.array([delayed_aif,delayed_vof])
    else: #We are only processing the AIF
        delayed_intensity = delay_sequence_padding(labels,delay_t)

    return delayed_volume, delayed_intensity

def early_bolus(volume_sequence, labels, delay_t=None):
    early_volume = np.zeros(volume_sequence.shape)
    nb_timepoints = volume_sequence.shape[-1]
    labels_shape = np.array(labels).shape

    last_volume = volume_sequence[:,:,:,-1]
    if delay_t == None:
        delay_t = random.randint(1,int(nb_timepoints/3))
    if delay_t == 0:
        return volume_sequence, labels
    
    early_volume[:,:,:,0:nb_timepoints-delay_t] = volume_sequence[:,:,:,delay_t:nb_timepoints]
    for i in range(nb_timepoints-delay_t,nb_timepoints):   
        early_volume[:,:,:,i] = volume_sequence[:,:,:,-1]
    if len(labels_shape)==2:#We are processing the AIF and the VOF
        early_aif = anticipate_sequence_padding(labels[0],delay_t)
        early_vof = anticipate_sequence_padding(labels[1],delay_t)
        early_intensity = np.array([early_aif,early_vof])
    else: #We are only processing the AIF
        early_intensity = anticipate_sequence_padding(labels,delay_t)

    return early_volume, early_intensity


class ISLES18DataGen_aifvof_otf(tf.keras.utils.Sequence): 
    def __init__(self, 
                 ctp_volumes,
                 annotations_aif,
                 annotations_vof,
                 minimum_number_volumes_ctp,
                 batch_size=1,
                 input_size=(256, 256, None,43),
                 time_arrival_augmentation = True,
                 delay_t = None,
                 shuffle=True,
                 normalize_hu=True,
                 scale_aif = True):
        self.ctp_volumes = ctp_volumes        
        self.labels_aif = annotations_aif
        self.labels_vof = annotations_vof
        self.minimum_number_volumes_ctp = minimum_number_volumes_ctp
        self.batch_size = batch_size
        self.delay_t = delay_t
        self.input_size = input_size
        self.shuffle = shuffle
        self.augment = time_arrival_augmentation
        self.n = len(self.ctp_volumes)
        self.indices = np.arange(len(self.ctp_volumes))
        self.normalize_hu = normalize_hu
        self.scale_aif = scale_aif
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __get_input(self, img_idx):
        #Get the volume
        
        fname = self.ctp_volumes[img_idx]['image']
        cur_nib = nib.load(fname)
        ctp_vals = cur_nib.get_fdata()
        volume_sequence = ctp_vals[:,:,:,0:self.minimum_number_volumes_ctp]
        if self.normalize_hu:
            volume_sequence = normalize(volume_sequence)
        #Get the labels
        case_id = self.ctp_volumes[img_idx]['image'].split('.')[-2]
        #print(case_id)
        label_aif = self.labels_aif[case_id]
        label_vof = self.labels_vof[case_id]
        if self.scale_aif:
            label_aif = normalize_zero_one(label_aif)
            label_vof = normalize_zero_one(label_vof)
        labels = [label_aif,label_vof]
        #labels = np.array([label_aif,label_vof])
        if self.augment:
            augment_functions = [early_bolus,late_bolus]
            random_augmentation = random.choice(augment_functions)
            if self.delay_t == None:
                self.delay_t = np.random.randint(0,10)
                #print("Voy a aumentar con un delaz de " + str(self.delay_t))
            volume, labels = random_augmentation(volume_sequence,[label_aif,label_vof], self.delay_t)
        return volume,labels

    
    def __getitem__(self, idx): #This function returns the batch 
        #print(self.indices)
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        #print(inds)
        #batch_x = [self.ctp_volumes[index] for index in inds]
        #batch_y = self.annotations[inds]
        batch_x, batch_y = [], []
        for index in inds:
            x, y = self.__get_input(index)
            batch_x.append(x)
            batch_y.append(y)
        return np.array(batch_x), np.array(batch_y).squeeze()

    def __len__(self):
        return self.n // self.batch_size

#Dataset generator
class ISLES18DataGen_aifvof_aug(tf.keras.utils.Sequence):
  
    def __init__(self, 
                 ctp_volumes,
                 annotations_aif,
                 annotations_vof,
                 minimum_number_volumes_ctp,
                 batch_size=1,
                 input_size=(256, 256, None,43),
                 time_arrival_augmentation = True,
                 delay_t = None,
                 shuffle=True):
        self.ctp_volumes = ctp_volumes        
        self.labels_aif = annotations_aif
        self.labels_vof = annotations_vof
        self.minimum_number_volumes_ctp = minimum_number_volumes_ctp
        self.batch_size = batch_size
        self.delay_t = delay_t
        self.input_size = input_size
        self.shuffle = shuffle
        self.augment = time_arrival_augmentation
        self.n = len(self.ctp_volumes)
        self.indices = np.arange(len(self.ctp_volumes))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __get_input(self, img_idx):
        #Get the volume
        ctp_vals = self.ctp_volumes[img_idx]['ctpvals']
        volume_sequence = normalize(ctp_vals)
        #Get the labels
        case_id = self.ctp_volumes[img_idx]['image'].split('.')[-2]
        #print(case_id)

        label_aif = normalize_zero_one(self.labels_aif[case_id])
        label_vof = normalize_zero_one(self.labels_vof[case_id])
        labels = [label_aif,label_vof]
        #labels = np.array([label_aif,label_vof])
        if self.augment:
            augment_functions = [early_bolus,late_bolus]
            random_augmentation = random.choice(augment_functions)
            if self.delay_t == None:
                self.delay_t = np.random.randint(0,10)
                #print("Voy a aumentar con un delaz de " + str(self.delay_t))
            volume, labels = random_augmentation(volume_sequence,[label_aif,label_vof], self.delay_t)
        return volume,labels

    
    def __getitem__(self, idx): #This function returns the batch 
        #print(self.indices)
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        #print(inds)
        #batch_x = [self.ctp_volumes[index] for index in inds]
        #batch_y = self.annotations[inds]
        batch_x, batch_y = [], []
        for index in inds:
            x, y = self.__get_input(index)
            batch_x.append(x)
            batch_y.append(y)
        return np.array(batch_x), np.array(batch_y).squeeze()

    def __len__(self):
        return self.n // self.batch_size


#Dataset generator
class ISLES18DataGen_aifvof(tf.keras.utils.Sequence):

  
    def __init__(self, 
                 ctp_volumes,
                 annotations_aif,
                 annotations_vof,
                 minimum_number_volumes_ctp,
                 batch_size=1,
                 input_size=(256, 256, None,43),
                 shuffle=True):
        self.ctp_volumes = ctp_volumes 
        self.labels_aif = annotations_aif
        self.labels_vof = annotations_vof
        self.minimum_number_volumes_ctp = minimum_number_volumes_ctp
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle        
        self.n = len(self.ctp_volumes)
        self.indices = np.arange(len(self.ctp_volumes))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __get_input(self, img_idx):
        #Get the volume
        ctp_vals = self.ctp_volumes[img_idx]['ctpvals']
        volume = normalize(ctp_vals)
        #Get the labels
        case_id = ctp_volumes[img_idx]['image'].split('.')[-2]

        label_aif = normalize_aif(self.labels_aif[0][case_id])
        label_vof = normalize_aif(self.labels_vof[0][case_id])
        #labels = np.array([label_aif,label_vof])
        return volume,[label_aif,label_vof]
        #return volume,label_aif
    
    def __getitem__(self, idx): #This function returns the batch 
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        #print(inds)
        #batch_x = [self.ctp_volumes[index] for index in inds]
        #batch_y = self.annotations[inds]
        batch_x, batch_y = [], []
        for index in inds:            
            x, y = self.__get_input(index)
            batch_x.append(x)
            batch_y.append(y)
        return np.array(batch_x), np.array(batch_y).squeeze()

    def __len__(self):
        return self.n // self.batch_size
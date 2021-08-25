import nibabel as nib
import glob
import numpy as np
import os
from natsort import natsorted

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
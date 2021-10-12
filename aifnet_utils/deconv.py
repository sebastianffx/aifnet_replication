import glob
import scipy
import os,sys
import zipfile
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy import ndimage
from tensorflow import keras
from natsort import natsorted
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from aifnet_utils.preprocess import read_nifti_file, normalize_single_volume, normalize_aif, process_scan, normalize_zero_one, normalize_volumes_in_sequence
from aifnet_utils.data_loaders import ISLES18DataGen_aif, read_isles_volumepaths_from_file_otf, read_isles_annotations_from_file, ISLES18DataGen_aifvof_otf
from aifnet_utils.data_loaders import delay_sequence_padding, anticipate_sequence_padding, late_bolus, early_bolus
from aifnet_utils.results import plot_predictions
from aifnet_utils.losses import MaxCorrelation
from scipy import signal
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.linalg import convolution_matrix, toeplitz, circulant
from sklearn.linear_model import Ridge
from matplotlib import pyplot, image, transforms
from scipy import ndimage
from numpy import inf
import matplotlib.patches as patches
import random
keras.backend.set_image_data_format('channels_last')
#%matplotlib inline
#!pwd



def build_deconv_A_matrix(aif):
    A = np.multiply(np.tril(np.ones((aif.shape[0],aif.shape[0]))), circulant(aif))
    return A

def build_circular_matrix_implicit(aif, m=2):
    M = m*aif.shape[0]
    A_circular = np.zeros((M,M))    
    aif_hat = np.concatenate((aif,np.zeros(aif.shape[0])), axis = 0)
    
    for i in range(A_circular.shape[0]):
        for j in range(A_circular.shape[1]):
            if j<=i:
                A_circular[i,j] = aif_hat[i-j]
            if j>i:
                A_circular[i,j] = aif_hat[M+i-j]
    np.savetxt('A_circular.csv',delimiter=',',X=A_circular)
    return A_circular

def compute_pseudoinverse(A):
    return scipy.linalg.pinv2(A)

def svd_truncated(A,truncation_vals=3):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    S = np.diag(1/s)
    S[-truncation_vals:,-truncation_vals:] = 0
    return u,S,vh

def perform_deconvolution(perf_data,pseudo_inverse):
    residues = np.zeros((perf_data.shape)) #WxHxSlicesxTime
    CBFs = np.zeros((perf_data.shape[0],perf_data.shape[1],perf_data.shape[2]))
    for cur_slice in range(perf_data.shape[2]):
        for i in range(perf_data.shape[0]):
            for j in range (perf_data.shape[1]):
                voi = perf_data[i,j,cur_slice,:]
                residue_f = np.matmul(pseudo_inverse,voi.T)
                CBFs[i,j,cur_slice] = np.max(residue_f)
                residues[i,j,cur_slice,:] = residue_f
    return CBFs, residues

def perform_deconvolution_circular(perf_data,pseudo_inverse):
    residues = np.zeros(((perf_data.shape[0],perf_data.shape[1],perf_data.shape[2],2*perf_data.shape[3]))) #WxHxSlicesxTime
    CBFs = np.zeros((perf_data.shape[0],perf_data.shape[1],perf_data.shape[2]))
    for cur_slice in range(perf_data.shape[2]):
        for i in range(perf_data.shape[0]):
            for j in range (perf_data.shape[1]):
                voi = np.concatenate((perf_data[i,j,cur_slice,:],np.zeros(perf_data[i,j,cur_slice,:].shape[0])), axis = 0)   
                #print(voi.shape)
                residue_f = np.matmul(pseudo_inverse,voi.T)
                CBFs[i,j,cur_slice] = np.max(residue_f)
                residues[i,j,cur_slice,:] = residue_f[0:86]
    return CBFs, residues

def plot_cbf_map(train_datagen, selected_slice,example_id, return_array_only=False):
    #Gettring Ground truth CBF
    path_case = '/'.join(train_datagen.ctp_volumes[example_id]['image'].split('/')[0:-2])
    path_cbf = glob.glob(path_case+"/*CBF*/*nii")[0]
    gt_cbf = nib.load(path_cbf).get_fdata()
    #gt_cbf[gt_cbf >contrast_threshold] = 0
    gt_img = np.array(normalize_zero_one(gt_cbf[:,:,selected_slice])*255,dtype = 'uint8')
    #gt_img = ndimage.rotate(gt_img, 90)
    #gt_img = np.flip(gt_img,axis=1)
    if return_array_only:
        return gt_img
    else:
        plt.imshow(gt_img,cmap=plt.cm.jet,vmax=2.9)
        return gt_img
    


def plot_tmax_map(train_datagen, selected_slice,example_id,return_array_only=False):
    #Gettring Ground truth Tmax
    path_case = '/'.join(train_datagen.ctp_volumes[example_id]['image'].split('/')[0:-2])
    path_tmax = glob.glob(path_case+"/*Tmax*/*nii")[0]
    healty_tmax = nib.load(path_tmax).get_fdata()
    #print(healty_tmax.shape)

    healty_tmax[healty_tmax > 6] = 0

    #print(np.max(healty_tmax))
    gt_healty = np.array(normalize_zero_one(healty_tmax[:,:,selected_slice])*255,dtype = 'uint8')
    #gt_healty = ndimage.rotate(gt_healty, 90)
    #gt_healty = np.flip(gt_healty,axis=1)
    if return_array_only:
        return gt_healty
    else:
        plt.imshow(gt_healty,cmap=plt.cm.jet)
        return gt_healty


def plot_estimatedCBF_map(gt_cbf, CBF, gt_healty, selected_slice,normalize_healthy_tissue, return_array_only=False):
    #Gettring Ground truth Tmax
    mask_zeros = gt_cbf != 0
    mask_zeros = np.array(mask_zeros,dtype='int')
    mask_zeros.shape
    img = CBF[:,:,selected_slice]
    img = np.multiply(img,mask_zeros)

    img[img >= np.max(img)] = 0
    #print(np.max(img))
    #img[img >contrast_threshold] = 0

    #estimated_cbf = np.array(normalize_zero_one(img)*255,dtype = 'uint8')
    #estimated_cbf = np.array(normalize_zero_one(img)*255,dtype = 'uint8')
    #estimated_cbf = ndimage.rotate(estimated_cbf, 90)
    #estimated_cbf = ndimage.rotate(img, 90)
    #estimated_cbf = np.flip(estimated_cbf,axis=1)

    if normalize_healthy_tissue:
        mask_healthy = gt_healty>0
        mask_healthy.shape
        mean_cbf_healthy = np.mean(img[mask_healthy])
        #print(mean_cbf_healthy)
        estimated_cbf = np.multiply(1/(mean_cbf_healthy), img)    
    if return_array_only:
        return estimated_cbf
    else:
        plt.imshow(estimated_cbf,cmap=plt.cm.jet)
        return estimated_cbf 



#Two ways of computing the psuedoinverse of A: 1) using a truncated SVD or using minsquares with Tikhonov regularization
def aif_inverse(A,number_of_truncated_values=8,epsilon=None):

    u,S,vh = svd_truncated(A,truncation_vals=number_of_truncated_values)
    max_S = np.max(np.diag(S))
    
    if epsilon != None:
        S[S<epsilon*max_S] = 0
    
    invD = np.matmul(vh,compute_pseudoinverse(S))
    invD = np.matmul(invD,u.T)
    return invD, S



def mask_CBF(gt_cbf,CBF_volume, selected_slice,gt_healty=None):
    mask_zeros = gt_cbf != 0
    mask_zeros = np.array(mask_zeros,dtype='int')
    mask_zeros.shape
    img = CBF_volume[:,:,selected_slice]
    img = np.multiply(img,mask_zeros)
    img[img >= np.max(img)] = 0
    estimated_cbf = img
    if type(gt_healty)=='numpy.ndarray':
        mask_healthy = gt_healty>0
        mask_healthy.shape
        mean_cbf_healthy = np.mean(img[mask_healthy])
        #print(mean_cbf_healthy)
        estimated_cbf = np.multiply(1/(mean_cbf_healthy), img)    
    return estimated_cbf
    
    

def save_nifti_from_array_and_referenceHeader(volume, reference_path, filename):
    reference_nib =  nib.load(reference_path)
    new_header = header=reference_nib.header.copy()
    new_nib = nib.Nifti1Image(volume,reference_nib.affine,header=new_header)
    new_nib.get_data_dtype() == np.dtype(np.float64)
    new_nib.header.get_xyzt_units()
    new_nib.header.set_data_dtype = reference_nib.header.get_data_dtype()
    new_nib.header.set_qform = reference_nib.header.get_qform()
    new_nib.header.set_zooms = reference_nib.header.get_zooms()
    nib.save(new_nib, filename)
    return
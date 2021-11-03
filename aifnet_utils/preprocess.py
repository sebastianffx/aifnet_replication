import numpy as np
import nibabel as nib

def read_nifti_file(filepath, min_num_sequence):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()[:,:,:,0:min_num_sequence]
    return scan

def normalize_volumes_in_sequence(volume_seq):
    norm_vol = np.zeros(volume_seq.shape)
    for vol_num in range(volume_seq.shape[-1]):
        norm_vol[:,:,:,vol_num] = normalize_single_volume(volume_seq[:,:,:,vol_num])
    return norm_vol

def normalize_single_volume(volume):
    """Normalize the volume"""
    min = 0
    max =   720 #Check this with Richard or Roland
    #skull_val = 1300
    volume[volume < min] = min
    #volume[volume > skull_val] = 0
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def normalize_aif(aif):
    normalized_aif = aif/np.max(aif)
    return normalized_aif

def normalize_zero_one(x):
    normalized_zero_one = (x-np.min(x))/(np.max(x)-np.min(x))
    return normalized_zero_one
    

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

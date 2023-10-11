import numpy as np
from tqdm import tqdm
import os
from PIL import Image


def create_the_masks(path_fixation_folder: str = None, folders_of_interest: list = None):
    """
    master function to create the combined match for all the folders located in path_fixation_folder

    Parameters
    ----------
    path_fixation_folder : str
        the path to the measurement folder
    folders_of_interest : list
        the list of the folders to be considered
        
    Returns
    ----------
    folder_of_interests : list
        the list of the folders considered
    """
    folders = os.listdir(path_fixation_folder)

    # if no folder of interest is provided, use the folders provided by the variable folders_of_interest
    if path_fixation_folder is None:
        assert folders_of_interest is not None
        folder_of_interests = folders_of_interest
        
    else:
        assert path_fixation_folder is not None
        
        # get the folders of interest (in a .txt file)
        folders_of_interest = load_folders()
        
        # add each measurement in folders_of_interest to the list of folders to be considered
        folder_of_interests = []
        for folder in folders:
            for measurement in os.listdir(os.path.join(path_fixation_folder, folder)):
                if measurement in folders_of_interest:
                    folder_of_interests.append(os.path.join(path_fixation_folder, folder, measurement))

    # create the GM / WM masks for each folder of interest
    for folder_of_interest in tqdm(folder_of_interests):
        _ = get_masks(folder_of_interest)
        
    return folder_of_interests
        
        
def load_folders():
    """
    load and returns the folders considered for the verification of the alignment protocol

    Returns
    -------
    folders_of_interest : list
        the list of the folders considered
    """
    dir_path = os.path.dirname(os.path.realpath(__file__).split('VerificationAlignment')[0])
    with open(os.path.join(dir_path, 'data', 'folders_verification.txt')) as f:
        lines = f.readlines()
    f.close()
    folders_of_interest = lines[0]
    return folders_of_interest



def get_masks(path: str = None, bg: bool = True):
    """
    obtain the masks (white matter, grey matter and background) by combining previsously manually drawn masks

    Parameters
    ----------
    path : str
        the path to the folder containing the annotations considered
    bg : bool
        boolean changing if WM or BG is the first class

    Returns
    -------
    masks
        the merged masks
    """
    BG = []
    WM = []
    path = os.path.join(path, 'annotation')
    
    # add the masks to the different lists
    for file in os.listdir(path):
        if '.txt' in file or 'noise' in file:
            pass
        else:
            im = Image.open(os.path.join(path, file))
            imarray = np.array(im)
            
            # add all the WM and BG files to the corresponding lists
            if 'BG' in file and not 'merged' in file:
                BG.append(imarray)
            elif 'WM' in file and not 'merged' in file and not 'GM' in file:
                WM.append(imarray)
            elif 'merged' in file or 'mask-viz' in file or 'mask_total' in file or 'ROI' in file or 'colored' in file:
                pass
            else:
                raise(NotImplementedError)
    
    # combine the WM and BG masks
    WM = combine_masks(WM)
    BG = combine_masks(BG)
    
    # return the merged masks
    return merge_masks(BG, WM, path, bg)


def combine_masks(masks: list):
    """
    combine previsously manually drawn masks

    Parameters
    ----------
    masks : list of array of shape (388,516)
        the manually drawn masks

    Returns
    -------
    base : array of shape (388,516)
        the combined mask
    """
    if masks:
        # use the first mask as a base
        base = masks[0]
        
        # for each of the mask, search for pixels equal to 255 and add them as positive values to the base mask
        for id_, mask in enumerate(masks[1:]):
            for idx, x in enumerate(mask):
                for idy, y in enumerate(x):
                    if base[idx, idy] == 255 or y == 255:
                        base[idx, idy] = 255
    
    # if no mask is found, everything is set to 0
    else:
        base = np.zeros((388, 516))
    return base


def merge_masks(BG, WM, path, bg):
    """
    merge masks is used to merge the previsously combined manually drawn masks

    Parameters
    ----------
    BG : array of shape (388,516)
        the combined background mask
    WM : array of shape (388,516)
        the combined white matter mask
    path : str
        the path to the folder containing the annotations considered
    bg : bool
        boolean changing if WM or BG is the first class

    Returns
    -------
    BG_merged : array of shape (388,516)
        the merged background mask
    GM_merged : array of shape (388,516)
        the merged grey matter mask
    WM_merged : array of shape (388,516)
        the merged white matter mask
    all_merged : array of shape (388,516)
        the three masks combined (one color = one class)
    """
    WM_merged = np.zeros(WM.shape)
    BG_merged = np.zeros(WM.shape)
    GM_merged = np.zeros(WM.shape)
    all_merged = np.zeros(WM.shape)

    # iterate through the pixels of the masks
    for idx, x in enumerate(WM):
        for idy, y in enumerate(x):
            
            if bg:
                # 1. check if the pixel is white matter
                if WM[idx, idy] == 255:
                    WM_merged[idx, idy] = 255
                    all_merged[idx, idy] = 255
                    
                # 2. if not, check if it is background
                elif BG[idx, idy] == 255:
                    BG_merged[idx, idy] = 255
                    all_merged[idx, idy] = 0
                
                # 3. if not, it is grey matter
                else:
                    GM_merged[idx, idy] = 255
                    all_merged[idx, idy] = 128
            
            else:
                # 1. check if it is background
                if WM[idx, idy] == 255:
                    WM_merged[idx, idy] = 255
                    all_merged[idx, idy] = 255

                # 2. if not, check if the pixel is white matter
                elif BG[idx, idy] == 255:
                    BG_merged[idx, idy] = 255
                    all_merged[idx, idy] = 0
                    
                # 3. if not, it is grey matter
                else:
                    GM_merged[idx, idy] = 255
                    all_merged[idx, idy] = 128

    # save the masks
    save_image(path, WM_merged, 'WM_merged')
    save_image(path, BG_merged, 'BG_merged')
    save_image(path, GM_merged, 'GM_merged')
    
    new_p = Image.fromarray(all_merged)
    if new_p.mode != 'L':
        new_p = new_p.convert('L')
        new_p.save(os.path.join(path, 'merged.jpeg'))
        new_p.save(os.path.join(path, 'merged.png'))
        
    return BG_merged, WM_merged, GM_merged, all_merged


def save_image(path: str, img: np.array, name: str):
    """
    save_image is used to save an image as a .png file

    Parameters
    ----------
    path : str
        the path to the folder containing the annotations considered
    img : array of shape (388,516)
        the image to be saved
    name : str
        the name of the image
    """
    new_p = Image.fromarray(img)
    if new_p.mode != 'L':
        new_p = new_p.convert('L')
        new_p.save(os.path.join(path, name + '.png'))
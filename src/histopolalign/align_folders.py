import numpy as np
from PIL import Image
import os, shutil
import datetime
import time
import matlab.engine
import scipy
import pickle
from scipy.io import savemat
from tqdm import tqdm

from histopolalign.prepare_images import FolderAlignHistology, count_pixels
from histopolalign.helpers import load_color_maps
from histopolalign.prepare_images import process_image_pathology


def create_align_folders(alignment_measurements: list, Verbose: bool = False):
    """
    create_align_folders is the master function calling create_the_alignment_folder for each measurement in the list

    Parameters
    ----------
    alignment_measurements : list
        the list of measurements to be aligned
    Verbose : bool, default is False
        if True, print the time taken by the function (default is False)
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        shutil.rmtree(os.path.join(dir_path, 'alignment', 'to_align'))
    except FileNotFoundError:
        pass

    for measurement in tqdm(alignment_measurements) if Verbose else alignment_measurements:
        create_the_alignment_folder(measurement, Verbose = Verbose)
        
    
def create_the_alignment_folder(measurement: FolderAlignHistology, Verbose: bool = False):
    """
    create_the_alignment_folder save the images in the polarimetry folder and creates the folder that will be used for the alignment later on

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement 
    Verbose : bool
        if True, print the time taken by the function (default is False)
    """     
    start = time.time()
    # apply the pre-recorded transformation to the histology images
    images = process_image_pathology(measurement.histology_cropped, measurement.labels_cropped, measurement.labels_GM_WM_cropped, measurement.center_of_mass, 
                                     angle = measurement.angle, flip = measurement.flip, shrink = measurement.shrink, x_offset = measurement.x_offest,
                                     y_offset = measurement.y_offest, selection = True)
    
    img = np.array(images[1])
    signal = np.where(np.sum(np.array(images[2]), axis = 2) != 0)
    for idx, idy in zip(signal[0], signal[1]):
        if sum(img[idx, idy]) == 0:
            img[idx,idy] = [255,255,255]
    
    measurement.registration_img = Image.fromarray(img.astype(np.uint8))
    measurement.registration_labels_img = images[2]
    measurement.registration_labels_GM_WM_img = images[3] 
    end = time.time()
    if Verbose:
        print("Process the images to apply the selected changes: {:.3f} seconds.".format(end - start))
        
    start = time.time()
    # save all the images 
    measurement.path_histology_polarimetry = os.path.join(measurement.folder_path, 'histology')
    save_the_images_into_polarimetry_folders(measurement)
    end = time.time()
    if Verbose:
        print("Save the processed images: {:.3f} seconds.".format(end - start))
    
    prepare_the_alignment_folder(measurement)
    

def save_the_images_into_polarimetry_folders(measurement: FolderAlignHistology):
    """
    save all the images in the given folder

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement 
    """
    path_pol_save = measurement.path_histology_polarimetry
    
    # save the registration, labels and labels_GM_WM images
    measurement.registration_img.save(os.path.join(path_pol_save, 'registration.png'))
    measurement.registration_labels_img.save(os.path.join(path_pol_save, 'labels.png'))
    measurement.registration_labels_GM_WM_img.save(os.path.join(path_pol_save, 'labels_GM_WM.png'))
    

def prepare_the_alignment_folder(measurement: FolderAlignHistology):
    """
    creates the folder that will be used for the alignment later on for one measurement

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    """
    # create the name and path of the folder
    now = datetime.datetime.now()
    fname = measurement.folder_path.split('\\')[-1]
    dt_string = fname +'__' + now.strftime("%d/%m/%Y %H:%M:%S").replace(' ', '_').replace('/', '_').replace(':', '_')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_name = os.path.join(dir_path, 'alignment', 'to_align/' + dt_string)
    
    try:
        os.mkdir(os.path.join(dir_path, 'alignment'))
    except FileExistsError:
        pass
    
    try:
        os.mkdir(os.path.join(dir_path, 'alignment', 'to_align'))
    except FileExistsError:
        pass
    
    os.mkdir(folder_name)
    os.mkdir(os.path.join(folder_name, 'propagation_msk'))
    measurement.to_align_link = folder_name


    # save the images in the alignment folder
    shutil.copyfile(os.path.join(measurement.polarimetry_path_gs.split('550nm')[0], '550nm', 'Depolarization_img.png'), os.path.join(folder_name, 'depolarization.png'))
    measurement.registration_img.save(os.path.join(folder_name, 'histology_rgb_upscaled.png'))
    measurement.registration_labels_img.save(os.path.join(folder_name, 'histology_labels_upscaled.png'))
    measurement.registration_labels_GM_WM_img.save(os.path.join(folder_name, 'histology_labels_GM_WM_upscaled.png'))
    measurement.img_polarimetry_gs.resize((516, 388), 
                            Image.Resampling.NEAREST).save(os.path.join(folder_name, 'polarimetry.png'))
    measurement.img_polarimetry_gs_650.resize((516, 388), 
                            Image.Resampling.NEAREST).save(os.path.join(folder_name, 'polarimetry_650.png'))
    
    
def semi_automatic_processing(alignment_measurements: list):
    """
    semi_automatic_processing is called to ask the pairs of landmarks points to the user using matlab
    
    Parameters
    ----------
    alignment_measurements : list
        the list of FolderAlignHistology containing the information and the images about the measurements
    """
    # check if one matlab engine should be started or not
    process_need = False
    for measurement in alignment_measurements:
        positions = get_positions(measurement)
        if os.path.exists(os.path.join(measurement.path_histology_polarimetry, 'mp_fp.pickle')):
            with open(os.path.join(measurement.path_histology_polarimetry, 'mp_fp.pickle'), 'rb') as handle:
                mp_fp = pickle.load(handle)
                
            try:
                mp_fp[frozenset(positions.items())]
            except KeyError:
                process_need = True
                
        else:
            process_need = True
    
    # if needed, start the matlab engine
    if process_need:
        eng = matlab.engine.start_matlab()
    
    try:
        shutil.rmtree('./temp')
    except FileNotFoundError:
        pass
    
    try:
        os.mkdir('./temp')
    except FileExistsError:
        pass
    
    # iterate over the measurements
    for measurement in alignment_measurements:

        if os.path.exists('./temp/mp_fp.mat'):
            os.remove('./temp/mp_fp.mat')
        if os.path.exists('./temp/mp_fp.pickle'):
            os.remove('./temp/mp_fp.pickle')
        
        with open('./temp/folder_aligned.txt', 'w') as f:
            f.write(measurement.to_align_link)
        with open('./temp/labels_number.txt', 'w') as f:
            f.write(str(measurement.labels_number))
        with open('./temp/labels_number_lab.txt', 'w') as f:
            f.write(str(measurement.labels_number_lab))
        with open('./temp/labels_number_lab_GM_WM.txt', 'w') as f:
            f.write(str(measurement.labels_number_lab_GM_WM))

        if os.path.exists(os.path.join(measurement.path_histology_polarimetry, 'mp_fp.pickle')):
            positions = get_positions(measurement)
            with open(os.path.join(measurement.path_histology_polarimetry, 'mp_fp.pickle'), 'rb') as handle:
                mp_fp = pickle.load(handle)
            try:
                savemat('./temp/mp_fp.mat', mp_fp[frozenset(positions.items())])
            except KeyError:
                pass
        else:
            pass
        
        # ask the user to select the landmarks points if needed
        if os.path.exists('./temp/mp_fp.mat'):
            shutil.copyfile('./temp/mp_fp.mat', os.path.join(measurement.to_align_link, 'mp_fp.mat'))
        else:
            eng.registration(nargout = 0)

    
        mp_fp = {}
        mp_fp['mp'] = scipy.io.loadmat(os.path.join(measurement.to_align_link, 'mp_fp.mat'))['mp']
        mp_fp['fp'] = scipy.io.loadmat(os.path.join(measurement.to_align_link, 'mp_fp.mat'))['fp']
        
        try:
            with open(os.path.join(measurement.path_histology_polarimetry, 'mp_fp.pickle'), 'rb') as handle:
                mp_fp_all = pickle.load(handle)
        except FileNotFoundError:
            mp_fp_all = {}
        mp_fp_all[frozenset(positions.items())] = mp_fp
        
        with open(os.path.join(measurement.to_align_link, 'mp_fp.pickle'), 'wb') as handle:
            pickle.dump(mp_fp_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # close the matlab engine if needed
    if process_need:
        eng.quit()
        
        
def get_positions(measurement: FolderAlignHistology):
    """
    get_positions returns the parameters for the re-alignement tool of the histology images

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement

    Returns
    -------
    positions : dict
        the parameters for the re-alignement tool of the histology images
    """
    positions = {}
    positions['angle'] = measurement.angle 
    positions['rotation'] = measurement.flip
    positions['shrink'] = measurement.shrink
    positions['x_offest'] = measurement.x_offest
    positions['y_offest'] = measurement.y_offest
    return positions
    
    
def create_propagation_mask(image: Image, folder_name: str, GM_WM: bool = False):
    """
    create_propagation_mask changes the color format from RGB to grayscale (maps RGB color to uint values)

    Parameters
    ----------
    image : Pillow image
        the label image to be converted as a grayscale
    folder_name : str
        the path to the folder in the to_align subfolder
    GM_WM : bool
        if True, the image is a GM_WM image (default is False)
    """
    color_code = load_color_maps()
    image_array = np.array(image)
    mask_propagation = np.zeros(image_array.shape[:2])
    
    # create a dictionary with the RGB color as key in the format of tuple and the uint value as value
    _, keys = count_pixels(image_array)
    color_code_tupled = {}
    for _, val in color_code.items():
        if tuple(val['RGB']) in keys:
            color_code_tupled[tuple(val['RGB'])] = val['code']
            
    mask = np.where(np.array(image_array).sum(axis = 2) != 0)
    for idx, idy in zip(mask[0], mask[1]):
        try:
            tuple(image_array[idx, idy])
            # mask_propagation[idx, idy] = color_code_tupled[tuple(image_array[idx, idy])]
        except:
            mask_propagation[idx, idy] = 0
        
    # create the image and save it  
    mask_prop = Image.fromarray(mask_propagation).convert('L')
    mask_path = os.path.join(folder_name, 'mask')
    try:
        os.mkdir(mask_path)
    except FileExistsError:
        pass
    if GM_WM:
        mask_prop.save(os.path.join(mask_path, 'mask_GM_WM.png'))
    else:
        mask_prop.save(os.path.join(mask_path, 'mask.png'))
from PIL import Image
import os
import numpy as np
import cv2
import numpy as np
from tqdm import tqdm
import time
from histopolalign.AlignImages.helpers import load_labels_idx, load_combined_img_link


def create_to_align_folder():
    """
    creates the folder to_align in the alignment folder

    Parameters
    ----------

    Returns
    -------
    """
    try:
        os.rmdir('./alignment/to_align')
    except:
        pass
    try:
        os.mkdir('./alignment/to_align')
    except:
        pass



def load_the_images(histology_path: str, Verbose: bool = False):
    """
    load all the images from the histology annotations (i.e. the 8 masks corresponding to the 8 classes)

    Parameters
    ----------
    histology_path : str
        the path to the directory containing the histology folders
    Verbose : bool
        whether to print the time it takes to load the images (default is False)

    Returns
    -------
    imgs_all : dict
        the dictionary containing the images for each histology folder
    """
    imgs_all = {}
    
    # get the folder names for the different histology annotations as well as a dictionnary linking the labels to indices
    filenames_histology = get_filenames_histology(histology_path)
    labels_idx = load_labels_idx()

    # iterate over the different histology folders
    start = time.time()
    for file_histology in (tqdm(filenames_histology) if Verbose else filenames_histology):
        imgs = {}
        
        # iterate over the different images in the histology folder
        for img_path in os.listdir(file_histology):
            
            # try to rename the image if it contains _HE in the file name
            try:
                os.rename(os.path.join(file_histology, img_path), os.path.join(file_histology, img_path.replace('_HE', '')))
                img_path =  img_path.replace('_HE', '')
            except PermissionError:
                pass
            
            # if we have an image, we load it and add it to the dictionary
            if '.png' in img_path:
                for key, val in labels_idx.items():
                    if key in img_path:
                        img = cv2.imread(os.path.join(file_histology, img_path), 0)

                        # taking a matrix of size 5 as the kernel to perform erosion and dilation to remove part of the noise
                        kernel = np.ones((5, 5), np.uint8)
                        img_erosion = cv2.erode(img, kernel, iterations=3)
                        img_dilation = cv2.dilate(img_erosion, kernel, iterations=3)
                        img_dilation[img_dilation == 0] = 1
                        img_dilation[img_dilation == 255] = 0
                        
                        # we add the image to the dictionary (and we multiply by the indice corresponding to the label)
                        img_dilation = val * (img_dilation != 0)
                        imgs[val] = np.asarray(img_dilation)
        
        # add the dictionary to the dictionary containing the images for all folders
        imgs_all[file_histology] = imgs
        
    imgs_all_ = {}
    for key, val in imgs_all.items():
        try:
            os.rename(key, key.replace('_HE', ''))
        except PermissionError:
            pass
        imgs_all_[key.replace('_HE', '')] = val
    imgs_all = imgs_all_
    
    end = time.time()
    if Verbose:
        print("Load the images: {:.3f} seconds.".format(end - start))
        
    return imgs_all



def get_filenames_histology(histology_path: str):
    """
    get the names of all the folders containing the histology stainings and annotations

    Parameters
    ----------
    histology_path : str
        the path to the directory containing the histology folders
    
    Returns
    -------
    fnames_histology : list
        the folder names in which histology annotations are stored
    """
    fnames_histology = []
    all_folders = os.listdir(histology_path)
    for folder in all_folders:
        for fol in os.listdir(os.path.join(histology_path, folder)):
            fnames_histology.append(os.path.join(histology_path, folder, fol))
    return fnames_histology



def get_combined_img(imgs_all: dict, force_recompute: bool = False, Verbose = False):
    """
    get_combined_img is the master function to combine the different masks for individual tissue types into a single image

    Parameters
    ----------
    imgs_all : dict
        the dictionary containing the images for each histology folder
    force_recompute : bool
        whether to recompute the combined images even if they already exist (default is False)
    Verbose : bool
        whether to print the time it takes to load the images (default is False)
    
    Returns
    -------
    """
    # iterate over the different histology folders
    start = time.time()
    for filename, imgs in imgs_all.items():
        
        # create the results directory in which the combined image will be stored
        results_directory = os.path.join(filename, 'results')
        try:
            os.mkdir(results_directory)
        except:
            pass
        
        start = time.time()
        
        # if the combined images already exist we do not process them again
        if os.path.exists(os.path.join(results_directory, 'combined_img.png')) and os.path.exists(os.path.join(results_directory, 'combined_img_WM_GM.png')) and not force_recompute:
            pass
        else:
            # if not, generate the combined images and convert the arrays into Image objects
            combined_image_RGB, combined_image_GM_WM_RGB = generate_combined_images(imgs)
            combined_image_RGB = Image.fromarray(combined_image_RGB)
            combined_image_GM_WM_RGB = Image.fromarray(combined_image_GM_WM_RGB)
        
            # save the images
            combined_image_RGB.save(os.path.join(results_directory, 'combined_img.png'))
            combined_image_GM_WM_RGB.save(os.path.join(results_directory, 'combined_img_WM_GM.png'))
            
    end = time.time()
    if Verbose:
        print("Get combined images: {:.3f} seconds.".format(end - start))



def generate_combined_images(imgs: list):
    """
    creates the combined images (combining the various masks) for the annotations of the tumor cell contents and the GW/WM mask

    Parameters
    ----------
    imgs : dict
        the dictionary containing the images for the different annotations for one histology folder

    Returns
    -------
    combined_image_RGB.astype(np.uint8), combined_image_GM_WM_RGB.astype(np.uint8): np.ndarray, np.ndarray
        the combined images for the annotations of the tumor cell contents and the GW/WM mask
    """
    # create the blank images for the annotations of the tumor cell contents and the GW/WM mask
    combined_image, combined_image_GM_WM = np.zeros(imgs[1].shape), np.zeros(imgs[1].shape)
    combined_image_RGB, combined_image_GM_WM_RGB = np.zeros((imgs[1].shape[0], imgs[1].shape[1], 3)), np.zeros((imgs[1].shape[0], imgs[1].shape[1], 3))
    
    # load the dictionnaries linking the labels to the RGB color values
    combined_img_RGB_link, combined_img_RGB_GM_WM_link = load_combined_img_link(), load_combined_img_link(GM_WM = True)

    # get the dictionnary linking the labels
    idx_labels = load_labels_idx(reverse = True)
    
    # create a dictionnary to link the labels to the RGB color values
    labels_RGB_link = {}
    labels_RGB_GM_WM_link = {}
    for idx_label, key in idx_labels.items():
        try:
            labels_RGB_link[idx_label] = combined_img_RGB_link[key]
        except:
            labels_RGB_GM_WM_link[idx_label] = combined_img_RGB_GM_WM_link[key]
    
    # iterate over all the different images
    for key, img in imgs.items():
        
        # if key < 7, we are dealing with the annotations of the tumor cell contents
        if key < 7:
            mask = 1 * (combined_image == 0)
            combined_image += np.multiply(mask,img)
        else:
            # else, we are dealing with the GW/WM mask
            mask = 1 * (combined_image_GM_WM == 0)
            combined_image_GM_WM += np.multiply(mask,img)

    # convert the images into the RGB images
    for idx, x in enumerate(combined_image):
        for idy, y in enumerate(x):
            try:
                combined_image_RGB[idx, idy] = labels_RGB_link[int(y)]
            except:
                pass
            try:
                combined_image_GM_WM_RGB[idx, idy] = labels_RGB_GM_WM_link[int(combined_image_GM_WM[idx, idy])]
            except:
                pass

    return combined_image_RGB.astype(np.uint8), combined_image_GM_WM_RGB.astype(np.uint8)
import json
import os
import functools
from collections import Counter
import numpy as np
from PIL import Image

def most_frequent(lst: list):
    """
    returns the most frequent element in a list

    Parameters
    ----------
    lst: list
        the list in which to find the most frequent element

    Returns
    -------
    most_frequent_element: object
        the most frequent element in the list
    """
    occurence_count = Counter(lst)
    most_frequent_element = occurence_count.most_common(1)[0][0]
    return most_frequent_element


def compare_two_lists(lst1: list, lst2: list):
    """
    compare two lists and check if they are equal

    Parameters
    ----------
    lst1, lst2 : list, list
        the lists to compare

    Returns
    -------
    bool
        whether the lists are equal or not
    """
    lst1.sort()
    lst2.sort()
    return functools.reduce(lambda x, y: x and y, map(lambda a, b: a == b, lst2, lst1), True)


def load_txt_file(filename: str):
    """
    load and returns the labels used for the annotations

    Parameters
    ----------
    filename : str
        the filename of the txt file to load
        
    Returns
    -------
    lines : list
        the loaded txt file
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', filename)) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '')
    return lines


def load_labels_idx(reverse: bool = False):
    """
    load and returns the color maps between the labels and the index

    Parameters
    ----------
    reverse : bool, optional
        indicates if the dictionnary map should be reversed, by default False
        
    Returns
    -------
    data : dict
        load and returns the color mapping between the labels and the index
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'labels_idx.json')) as json_file:
        data = json.load(json_file)
    if reverse:
        data = {v: k for k, v in data.items()}
    return data


def load_combined_img_link(GM_WM: bool = False):
    """
    load and returns the RGB color maps between the labels and the labels

    Parameters
    -------
    GM_WM : bool
        indicates wether the color code loaded should be the one of the GM / WM or the tumor cell content one
        
    Returns
    -------
    img_link : dict
        load and returns the color maps between the labels and the index
    """
    if GM_WM:
        fname = 'combined_img_RGB_GM_WM_link.json'
    else:
        fname = 'combined_img_RGB_link.json'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', fname)) as json_file:
        data = json.load(json_file)
    img_link = {}
    for lab, val in data.items():
        img_link[lab] = tuple(val)
    return img_link


def load_wavelengths():
    """
    load and returns the wavelengths usable by the IMP

    Returns
    -------
    wavelenghts : list
        the wavelengths usable by the IMP
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'wavelengths.txt')) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '') + 'nm'
    return lines


def load_color_maps(GM_WM: bool = False):
    """
    load and returns the RGB and index color maps

    Parameters
    -------
    GM_WM : bool
        indicates wether the color code loaded should be the one of the GM / WM or the tumor cell content one
        
    Returns
    -------
    data : dict
        the color maps
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if GM_WM:
        fname = 'colors_GM_WM.json'
    else:
        fname = 'colors.json'
    with open(os.path.join(dir_path, 'data', fname)) as json_file:
        data = json.load(json_file)
    return data


def load_color_code_links(GM_WM: bool = False):
    """
    get the mapping between the RGB values and the index values

    Parameters
    -------
    GM_WM : bool
        indicates wether the color code loaded should be the one of the GM / WM or the tumor cell content one
        
    Returns
    -------
    color_code_link : dict
        the mapping between the RGB values and the index values
    """
    color_code = load_color_maps(GM_WM = GM_WM)
    color_code_link = {}
    for _, codes in color_code.items():
        color_code_link[tuple(codes['RGB'])] = codes['code']
        color_code_link[(0, 0, 0)] = 7
    return color_code_link


def load_param_matching_pts():
    """
    get the parameters for the matching points with the skeletons

    Parameters
    -------
        
    Returns
    -------
    data : dict
        the parameters for the matching points with the skeletons
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'param_matching_pts.json')) as json_file:
        data = json.load(json_file)
    data['skeletonMatchCost'] = np.zeros(data['numIter'])
    data['affineCost'] = np.zeros(data['numIter'])
    data['bendingEnergy'] = np.zeros(data['numIter'])
    data['matchRatio'] = np.zeros(data['numIter'])
    return data


def get_folders_lacking_orientation(histology_path: str):
    """
    get the folders for which the pre-alignment has not been done

    Parameters
    -------
    histology_path : str
        the path to the folder containing the histology images and the annotations 
        
    Returns
    -------
    fnames_histology : list
        the list of all the folders
    fnames_histology_processed : list
        the list of the folders for which the pre-alignment has not been done
    """
    # load the fields required for the pre-alignment
    position_names = load_txt_file('position_names.txt')
    fnames_histology = []
    fnames_histology_processed = []
    
    # get the list of all the folders
    all_folders = os.listdir(histology_path)
    for folder in all_folders:
        for fol in os.listdir(os.path.join(histology_path, folder)):
            
            # check if the parameters for the pre-alignment have been obtained
            if os.path.exists(os.path.join(histology_path, folder, fol, 'annotation', 'correction_alignment.json')):
                f = open(os.path.join(histology_path, folder, fol, 'annotation', 'correction_alignment.json'))
                data = list(json.load(f).keys())

                # check if all the fields are present
                if compare_two_lists(position_names, data):
                    fnames_histology_processed.append(os.path.join(histology_path, folder, fol))
                else:
                    fnames_histology.append(os.path.join(histology_path, folder, fol))
            else:
                fnames_histology.append(os.path.join(histology_path, folder, fol))
                
    return fnames_histology, fnames_histology_processed


def overlay_imgs(background: np.array, overlay: np.array, param: float = 0.5):
    """
    overlay_imgs is used to overlay two RGB images

    Parameters
    -------
    background, overlay : np.array, np.array
        the background and overlay images
    param : float, optional
        the parameter used to blend the two images (default is 0.5)
        
    Returns
    -------
    fnames_histology : list
        the list of all the folders
    fnames_histology_processed : list
        the list of the folders for which the pre-alignment has not been done
    """
    # convert the images to RGBA
    background = Image.fromarray(background.astype(np.uint8)).convert("RGBA")
    overlay = Image.fromarray(overlay.astype(np.uint8)).convert("RGBA")

    # blend the two images
    new_img = Image.blend(background, overlay, param)
    return new_img


def save_imgs_alignment(measurement):
    """
    save_imgs_alignment is used to save the images of the histology and polarimetry in a temporary folder
    
    Parameters
    -------
    measurement : FolderAlignHistology
        the FolderAlignHistology object containing the information about the different measurements
        
    Returns
    -------
    """
    # get the path to the temporary folder
    dir_path = os.path.dirname(os.path.realpath(__file__)).split('src')[0]
        
    # save the image of histology
    path_img = os.path.join(dir_path, 'notebooks', 'temp', 'labels_GM_WM.png')
    Image.fromarray(measurement.labels_GM_WM_contour).save(path_img)
    path_img = os.path.join(dir_path, 'notebooks', 'temp', 'labels.png')
    Image.fromarray(measurement.labels_contour).save(path_img)
    path_img = os.path.join(dir_path, 'notebooks', 'temp', 'histology.png')
    Image.fromarray(measurement.histology_contour).save(path_img)
        
    # img_histology.save()
    path_img = os.path.join(dir_path, 'notebooks', 'temp', 'polarimetry.png')
    measurement.img_polarimetry_gs_650.save(path_img)
        
        
        
        
        
"""

def load_position_names():
    load and returns the labels used for the annotations

    Returns
    -------
    labels : list
        the labels used for the annotations
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'position_names.txt')) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '')
    return lines
        
"""
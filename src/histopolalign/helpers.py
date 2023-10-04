import json
import os
import functools
from collections import Counter
import numpy as np
from PIL import Image

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def compare_two_lists(lst1, lst2):
    lst1.sort()
    lst2.sort()
    return functools.reduce(lambda x, y: x and y, map(lambda a, b: a == b, lst2, lst1), True)

def load_labels():
    """
    load and returns the labels used for the annotations

    Returns
    -------
    labels : list
        the labels used for the annotations
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'labels.txt')) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '')
    return lines


def load_position_names():
    """
    load and returns the labels used for the annotations

    Returns
    -------
    labels : list
        the labels used for the annotations
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'position_names.txt')) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '')
    return lines


def load_labels_idx(reverse: bool = False):
    """
    load and returns the color maps between the labels and the index

    Returns
    -------
    idx_labels : dict
        load and returns the color maps between the labels and the index
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'labels_idx.json')) as json_file:
        data = json.load(json_file)
    if reverse:
        data = {v: k for k, v in data.items()}
    return data


def load_combined_img_link(GM_WM: bool = False):
    """
    load and returns the color maps between the labels and the index

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
    load and returns the color maps

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
    color_code = load_color_maps(GM_WM = GM_WM)
    color_code_link = {}
    for _, codes in color_code.items():
        color_code_link[tuple(codes['RGB'])] = codes['code']
        color_code_link[(0, 0, 0)] = 7
    return color_code_link


def load_param_matching_pts(reverse: bool = False):
    """"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'param_matching_pts.json')) as json_file:
        data = json.load(json_file)
    data['skeletonMatchCost'] = np.zeros(data['numIter'])
    data['affineCost'] = np.zeros(data['numIter'])
    data['bendingEnergy'] = np.zeros(data['numIter'])
    data['matchRatio'] = np.zeros(data['numIter'])
    return data


def get_folders_lacking_orientation(histology_path: str):
    position_names = load_position_names()
    fnames_histology = []
    fnames_histology_processed = []
    all_folders = os.listdir(histology_path)
    for folder in all_folders:
        for fol in os.listdir(os.path.join(histology_path, folder)):
            if os.path.exists(os.path.join(histology_path, folder, fol, 'annotation', 'correction_alignment.json')):
                f = open(os.path.join(histology_path, folder, fol, 'annotation', 'correction_alignment.json'))
                data = list(json.load(f).keys())

                if compare_two_lists(position_names, data):
                    fnames_histology_processed.append(os.path.join(histology_path, folder, fol))
                else:
                    fnames_histology.append(os.path.join(histology_path, folder, fol))
            else:
                fnames_histology.append(os.path.join(histology_path, folder, fol))
    return fnames_histology, fnames_histology_processed


def overlay_imgs(background, overlay, param = 0.5):
    background = Image.fromarray(background.astype(np.uint8)).convert("RGBA")
    overlay = Image.fromarray(overlay.astype(np.uint8)).convert("RGBA")

    new_img = Image.blend(background, overlay, param)
    return new_img


def save_imgs_alignment(measurement):
    
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
        
        

import os
from collections import defaultdict
import pickle
import numpy as np

from histopolalign.AlignImages.helpers import load_color_code_links
from histopolalign.AlignImages.prepare_images import FolderAlignHistology


def create_numerical_values(alignment_measurements: list):
    """
    create_numerical_values is the master function to extract the numerical values from the MM.npz file for the different tissue regions

    Parameters
    ----------
    alignment_measurements : list of FolderAlignHistology
        the list of the different FolderAlignHistology objects containing the information about the different measurements

    Returns
    -------
    """
    for measurement in alignment_measurements:
        # create the final labels combining the labels of TCC and GM/WM
        create_final_labels(measurement)
        measurement.path_numerical = os.path.join(measurement.path_histology_polarimetry, 'numerical')
        
        # get the numercial values for the different tissue regions
        _ = get_numerical_values(measurement, wavelength = '550nm')
        values = combine_the_results(measurement, wavelength = '550nm')
    return values

    
def create_final_labels(measurement: FolderAlignHistology):
    """
    create_final_labels creates the final labels combining the labels of TCC and GM/WM

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology object containing the information about the different measurements

    Returns
    -------
    """
    color_code_link = load_color_code_links()
    color_code_link_GM_WM = load_color_code_links(GM_WM = True)
    
    img_labels_propagated = measurement.labels_final
    img_labels_propagated_GM_WM = measurement.labels_GM_WM_final

    mask = np.where(np.logical_or(np.sum(img_labels_propagated, axis = 2) != 0, np.sum(img_labels_propagated_GM_WM, axis = 2) != 0))
    img_labels_final = np.zeros(np.asarray(img_labels_propagated).shape[0:2])
    for idx, idy in zip(mask[0], mask[1]):
        try:
            val = color_code_link[tuple(img_labels_propagated[idx, idy])]
        except:
            val = 0
        
        if val == 0:
            pass
        else:
            try:
                val = val + 10 * color_code_link_GM_WM[tuple(img_labels_propagated_GM_WM[idx, idy])]
            except:
                val = val
        
        img_labels_final[idx, idy] = val

    measurement.img_labels_final = img_labels_final


def get_numerical_values(measurement: FolderAlignHistology, wavelength: str = '550nm'):
    """
    get_numerical_values is the function used to extract the numerical values from the MM for the different tissue regions

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology object containing the information about the different measurements
    wavelength : str
        the wavelength of the measurement

    Returns
    -------
    [depolarization_values, retardance_values, azimuth_values] : list
        the list containing the different numerical values for the different tissue regions
    """
    # load the MM.npz file
    path_MM = os.path.join('\\'.join(measurement.polarimetry_path_gs.split('/')[:-1]), 'MM.npz')
    img_labels = measurement.img_labels_final
    path_numerical_results = measurement.path_numerical
    MM = np.load(path_MM) 
    
    # initialize the different dictionaries
    depolarization_values = defaultdict(list)
    retardance_values = defaultdict(list)
    azimuth_values = defaultdict(list)
    
    # extract the different numerical values for the different tissue regions
    dep = MM['totP']
    linR = MM['linR']
    azimuth = MM['azimuth']
    for idx, x in enumerate(img_labels):
        for idy, y in enumerate(x):
            if y != 0:
                depolarization_values[y].append(dep[idx, idy])
                retardance_values[y].append(linR[idx, idy])
                azimuth_values[y].append(azimuth[idx, idy])
                
    # save the numerical values in pickle files
    try:
        os.mkdir(path_numerical_results)
    except:
        pass
    with open(os.path.join(path_numerical_results, 'numerical_values ' + wavelength + '.pickle'), 'wb') as handle:
        pickle.dump([depolarization_values, retardance_values, azimuth_values], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return [depolarization_values, retardance_values, azimuth_values]
        
        
def combine_the_results(measurement: FolderAlignHistology, wavelength: str = '550nm'):
    """
    combine_the_results is used to combine the numerical values for the different tissue regions into the TCC mask only
    for the different measurements and save them in pickle files

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology object containing the information about the different measurements
    wavelength : str
        the wavelength of the measurement

    Returns
    -------
    res_combined: dict
        the dictionary containing the different numerical values for the different tissue regions for the TCC mask only
    """
    path_numerical_results = measurement.path_numerical

    # load the numerical values
    with open(os.path.join(path_numerical_results, 'numerical_values ' + wavelength + '.pickle'), 'rb') as handle:
        res = pickle.load(handle)
    
    RGB_link_image = {1: 'Tumor 0-30', 2: 'Tumor 30-70', 3: 'Tumor 70-100', 4: 'NormalTissue', 5: 'Fibrosis',
                    6: 'Necrosis', 0: 'Background'}
    RGB_link_image_GM_WM = {1: 'Grey Matter', 2: 'White Matter', 0: 'Background'}

    # initialize the different dictionaries
    res_combined = {}
    res_combined_GM_WM = {}

    # combine the numerical values for the different tissue regions into the TCC mask only
    for res_param, param in zip(res, ['Depolarization', 'Linear retardance']):
        
        res_param_combined = {}
        res_param_combined_GM_WM = {}

        # iterate over the different tissue regions
        for idx, ress in res_param.items():
            if RGB_link_image[idx % 10] in res_param_combined.keys():
                res_param_combined[RGB_link_image[idx % 10]] = res_param_combined[RGB_link_image[idx % 10]] + ress
            else:
                res_param_combined[RGB_link_image[idx % 10]] = ress

            try:
                if RGB_link_image_GM_WM[idx // 10] in res_param_combined_GM_WM.keys():
                    res_param_combined_GM_WM[RGB_link_image_GM_WM[idx // 10]] = res_param_combined_GM_WM[RGB_link_image_GM_WM[idx // 10]] + ress
                else:
                    res_param_combined_GM_WM[RGB_link_image_GM_WM[idx // 10]] = ress
            except:
                assert idx // 10 == 7

        res_param_combined_cleared = {}
        for key, val in res_param_combined.items():
            if len(val) > 100:
                res_param_combined_cleared[key] = val
        
        res_param_combined_GM_WM_cleared = {}
        for key, val in res_param_combined_GM_WM.items():
            if len(val) > 100:
                res_param_combined_GM_WM_cleared[key] = val
                
        res_combined[param] = res_param_combined_cleared
        res_combined_GM_WM[param] = res_param_combined_GM_WM_cleared

    # save the numerical values in pickle files
    with open(os.path.join(path_numerical_results, 'numerical_values_combined ' + wavelength + '.pickle'), 'wb') as handle:
        pickle.dump(res_combined, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return res_combined, res_combined_GM_WM
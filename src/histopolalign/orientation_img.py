import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import cv2
import json
from rembg import remove
import time
import traceback

from histopolalign.helpers import load_wavelengths, load_position_names, compare_two_lists
from histopolalign.semi_automatic_tool import process_image_pathology
from histopolalign import semi_automatic_tool




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





def get_the_images_and_center_of_mass(histology_path: str):
    """
    function used to rename the histology images filenames, to load them as well as to compute the center of mass of the histology image

    Parameters
    ----------
    histology_path : str
        the path to the folders containing the histology subfolders

    Returns
    -------
    images : dict
        the dictionnary containing the loaded images
    center_of_mass : dict
        the center of the histology images
    to_remove : list
        the folders for which the orientation dictionnary has already been obtained
    """
    fnames_histology, fnames_histology_processed = get_folders_lacking_orientation(histology_path)

    images = {}
    center_of_mass = {}
    
    for fname in tqdm(fnames_histology + fnames_histology_processed):
        new_path = rename_histology_files(histology_path, fname)
        for file in os.listdir(new_path):
            if file.endswith('.png'):
                if '-ds.png' in file:
                    fname_new = '-'
                else:
                    fname_new = '-labels'

                old_name = os.path.join(new_path, file)
                new_name = os.path.join(new_path, new_path.split('\\')[-1] + fname_new + file.split(fname_new)[-1])
                new_name = new_name.replace('_HE', '')
                try:
                    os.rename(old_name, new_name)
                except PermissionError:
                    pass
        fname = fname.replace('_HE', '')

        continue_ = True
        if continue_:
            path_img = os.path.join(new_path, fname.split('\\')[-1] + '-ds.png')
            # load the histology image
            images[new_path] = Image.open(path_img.replace('_HE', ''))

            # extract only the foreground
            x, y = get_center_of_mass(images[os.path.join(histology_path, fname)])
            center_of_mass[os.path.join(histology_path, fname)] = [x, y]

    return images, center_of_mass, fnames_histology + fnames_histology_processed




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










def load_images_from_dict(fnames_histology_link):
    """
    function to load the polarimetric, labels and histology images

    Parameters
    ----------
    fnames_histology_link : dict
        the dictionnary to link the histology to the polarimetric measurements

    Returns
    -------
    data : dict
        the dictionnary containing both the images and the paths to the images
    """
    data = {}
    for polarimetry, histology in fnames_histology_link.items():

        # load histology image
        histology_path = os.path.join(histology, histology.split('\\')[-1] + '-ds.png')
        img_histology = Image.open(histology_path)

        # path to the positions
        positions_path = os.path.join(histology, 'annotation/correction_alignment.json')
        
        # path to the mask for background foreground segmentation
        annotation_path = os.path.join(polarimetry, 'annotation/ROI.tif')
        
        # load labels image
        labels_path = os.path.join(histology, 'results', 'combined_img.png')
        img_labels = Image.open(labels_path)
        
        labels_GM_WM_path = os.path.join(histology, 'results', 'combined_img_WM_GM.png')
        img_labels_GM_WM = Image.open(labels_GM_WM_path)

        # load polarimetry images
        polarimetry_path = os.path.join(polarimetry, 'polarimetry/550nm/' + polarimetry.split('\\')[-1] +
                                                      '_550nm_realsize.png')
        img_polarimetry = Image.open(polarimetry_path).convert('L')
        
        polarimetry_path_650 = os.path.join(polarimetry, 'polarimetry/650nm/' + polarimetry.split('\\')[-1] +
                                                      '_650nm_realsize.png')
        img_polarimetry_650 = Image.open(polarimetry_path_650).convert('L')

        img_dic = {'polarimetry': img_polarimetry, 'polarimetry_650': img_polarimetry_650, 
                   'labels': img_labels, 'labels_GM_WM': img_labels_GM_WM,
                   'histology': img_histology}
        
        path_dic = {'polarimetry': polarimetry_path, 'polarimetry_650': polarimetry_path_650,
                    'labels': labels_path, 'labels_GM_WM': labels_GM_WM_path, 
                    'histology': histology_path, 'position': positions_path,
                   'annotation': annotation_path}
        
        data[polarimetry.split('\\')[-1]] = {'images': img_dic,
                                            'paths': path_dic}
    return data










def get_combined_data(center_of_mass: dict, histology_fnames_link: dict, data: dict):
   
    combined_data = {}
    
    for folder, _ in tqdm(center_of_mass.items(), total = len(center_of_mass)):
        data_folder = {}
        try:
            fol = histology_fnames_link[folder]
            img_histo = data[fol.split('\\')[-1]]['images']['histology']
            data_folder['histology'] = img_histo

            img_labels = data[fol.split('\\')[-1]]['images']['labels']
            data_folder['labels'] = img_labels

            img_labels_GM_WM = data[fol.split('\\')[-1]]['images']['labels_GM_WM']
            data_folder['labels_GM_WM'] = img_labels_GM_WM

            path = '\\'.join(data[fol.split('\\')[-1]]['paths']['histology'].split('\\')[:-1])

            img_registration = process_image_pathology(img_histo, img_labels, path, center_of_mass[folder], angle = 0, image_labels_GM_WM = img_labels_GM_WM,
                                                       flip = '')

            data_folder['histology_cropped'] = img_registration[1]
            data_folder['labels_cropped'] = img_registration[2]
            data_folder['labels_GM_WM_cropped'] = img_registration[3]


            combined_data[folder] = data_folder
        except:
            traceback.print_exc()
            
    return combined_data






def record_the_transformations(combined_data: dict, labels_number: dict, data: dict, histology_fnames_link: dict):
    parameters = {}
    for folder, imgs in combined_data.items():

        if labels_number[folder] == 1:
            parameters[folder] = [0, 'n']

        else:

            find_params = True

            while find_params:

                img_polarimeter = data[histology_fnames_link[folder].split('\\')[-1]]['images']['polarimetry']
                img_polarimeter_650 = data[histology_fnames_link[folder].split('\\')[-1]]['images']['polarimetry_650']

                cv2.imshow("Histology", np.asarray(imgs['histology_cropped']))
                cv2.imshow("Polarimetry 550nm", np.asarray(img_polarimeter))
                cv2.imshow("Polarimetry 650nm", np.asarray(img_polarimeter_650))
                new_img = Image.blend(imgs['histology_cropped'], img_polarimeter.convert('RGB'), 0.8)
                cv2.imshow("Overlaid 550nm", np.asarray(new_img))
                new_img_650 = Image.blend(imgs['histology_cropped'], img_polarimeter_650.convert('RGB'), 0.8)
                cv2.imshow("Overlaid 650nm", np.asarray(new_img_650))

                cv2.waitKey(0)
                cv2.destroyAllWindows()

                angle = input("Enter angle: ")
                angle = int(angle)
                flip = input("Should image be flipped [v/h/n]: ")

                rotated = imgs['histology_cropped'].rotate(angle, expand=True)
                rotated = rotated.crop(box=(rotated.size[0]/2 - imgs['histology_cropped'].size[0]/2,
                       rotated.size[1]/2 - imgs['histology_cropped'].size[1]/2,
                       rotated.size[0]/2 + imgs['histology_cropped'].size[0]/2,
                       rotated.size[1]/2 + imgs['histology_cropped'].size[1]/2))

                if flip == 'vh' or flip == 'hv':
                    img_new = rotated.transpose(method=Image.FLIP_TOP_BOTTOM)
                    img_new = img_new.transpose(method=Image.FLIP_LEFT_RIGHT)
                elif flip == 'v':
                    img_new = rotated.transpose(method=Image.FLIP_TOP_BOTTOM)
                elif flip == 'h':
                    img_new = rotated.transpose(method=Image.FLIP_LEFT_RIGHT)
                else:
                    img_new = rotated

                new_img = Image.blend(img_new, img_polarimeter.convert('RGB'), 0.8)
                cv2.imshow("Overlaid", np.asarray(new_img))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                inputed = input("Wanna re-do it? [y/n]")
                find_params = inputed == 'y'

                parameters[folder] = [angle, flip]
    return parameters


def save_the_rotations(parameters:dict, center_of_mass: dict):
    for folder, vals in parameters.items():
        positions = {}
        positions['x'] = center_of_mass[folder][1]
        positions['y'] = center_of_mass[folder][0]
        positions['angle'] = vals['angle']
        positions['rotation'] = vals['flip']
        positions['shrink'] = vals['shrink']
        positions['x_offest'] = vals['x_offest']
        positions['y_offest'] = vals['y_offest']

        json_string = json.dumps(positions)

        try:
            os.mkdir(os.path.join(folder, 'annotation'))
        except:
            pass

        with open(os.path.join(folder, 'annotation', 'correction_alignment.json'), 'w') as outfile:
            outfile.write(json_string)
            
            
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

            

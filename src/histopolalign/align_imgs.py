import numpy as np
from PIL import Image
import os, shutil
import cv2
import skimage
import matplotlib.pyplot as plt
import time
import pickle
import scipy.io as sio
from tqdm import tqdm
import skimage.morphology as morphology

from histopolalign.prepare_images import FolderAlignHistology
from histopolalign.match_skeletons import warp_img
from histopolalign.helpers import load_color_code_links, most_frequent, load_color_maps
from histopolalign.align_folders import get_positions


def align_img_master(alignment_measurements):
    dir_path = os.path.dirname(os.path.realpath(__file__)).split('src')[0]
    path_temp = os.path.join(dir_path, 'notebooks', 'temp')
    
    try:
        shutil.rmtree(path_temp)
    except FileNotFoundError:
        pass
    try:
        os.mkdir(path_temp)
    except FileExistsError:
        pass
    
    with open(os.path.join(path_temp, 'align_measurement.pickle'), 'wb') as handle:
        pickle.dump(alignment_measurements, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    os.system("python " + os.path.join(dir_path, 'src', 'histopolalign', 'imgJ_align.py'))
    
    with open(os.path.join(path_temp, 'align_measurement.pickle'), 'rb') as handle:
        alignment_measurements = pickle.load(handle)
    
    try:
        shutil.rmtree(path_temp)
        pass
    except FileNotFoundError:
        pass
    
    return alignment_measurements
    
    
def align_w_imageJ(ij, alignment_measurements):
    current_folder = '"' + os.path.dirname(os.path.realpath(__file__)).split('src')[0].replace('\\', '/')
    current_folder_no_start = current_folder.replace('"', '')
    
    macro = """
    open(""" + current_folder + """/notebooks/temp/histology.png");
    open(""" + current_folder + """/notebooks/temp/labels.png");
    open("""  + current_folder + """/notebooks/temp/labels_GM_WM.png");
    open(""" + current_folder + """/notebooks/temp/polarimetry.png");

    call("bunwarpj.bUnwarpJ_.loadLandmarks", """ + current_folder + """/notebooks/temp/coordinates.txt");
    run("bUnwarpJ", "load=""" + current_folder_no_start + """/notebooks/temp/coordinates.txt source_image=polarimetry.png target_image=labels_GM_WM.png registration=Accurate image_subsample_factor=0 initial_deformation=Coarse final_deformation=[Very Fine] divergence_weight=0 curl_weight=0 landmark_weight=8 image_weight=0 consistency_weight=2 stop_threshold=0.01");
    saveAs("Tiff", """ + current_folder + """/notebooks/temp/labels_GM_WM_registered.tif");
    close();
    close();
    
    call("bunwarpj.bUnwarpJ_.loadLandmarks", """ + current_folder + """/notebooks/temp/coordinates.txt");
    run("bUnwarpJ", "load=""" + current_folder_no_start + """/notebooks/temp/coordinates.txt source_image=polarimetry.png target_image=labels.png registration=Accurate image_subsample_factor=0 initial_deformation=Coarse final_deformation=[Very Fine] divergence_weight=0 curl_weight=0 landmark_weight=8 image_weight=0 consistency_weight=2 stop_threshold=0.01");
    saveAs("Tiff", """ + current_folder + """/notebooks/temp/labels_registered.tif");
    close();
    close();

    call("bunwarpj.bUnwarpJ_.loadLandmarks", """ + current_folder + """/notebooks/temp/coordinates.txt");
    run("bUnwarpJ", "load=""" + current_folder_no_start + """/notebooks/temp/coordinates.txt source_image=polarimetry.png target_image=histology.png registration=Accurate image_subsample_factor=0 initial_deformation=Coarse final_deformation=[Very Fine] divergence_weight=0 curl_weight=0 landmark_weight=4 image_weight=0 consistency_weight=2 stop_threshold=0.01");
    saveAs("Tiff", """ + current_folder + """/notebooks/temp/histo_registered.tif");

    close();
    close();
    close();
    close();
    close();
    close();
    """
    
    print(macro)
    
    for measurement in alignment_measurements:
        write_mp_fp(measurement)
        save_imgs_alignment(measurement)
        ij.py.run_macro(macro)
        align_images_imgj(measurement)
        save_aligned_images(measurement)
        move_the_alignment_results(measurement)


def save_imgs_alignment(measurement):
    
    dir_path = os.path.dirname(os.path.realpath(__file__)).split('src')[0]
    try:
        os.mkdir(os.path.join(dir_path, 'notebooks', 'temp'))
    except:
        pass
    
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
    
def get_mp_fp(measurement: FolderAlignHistology):
    """
    align_images_py align the images using the points in the mp_fp.mat file

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    """
    # TODO: check if this is correct
    # loads the matched landmark points
    mp_fp_name = os.path.join(measurement.to_align_link, 'mp_fp.pickle')
    mp_fp = {}
    mp_fp_mat = sio.loadmat(mp_fp_name.replace('pickle', 'mat'))
    mp_fp[frozenset(get_positions(measurement).items())] = {'mp': mp_fp_mat['mp'], 'fp': mp_fp_mat['fp']}
    with open(mp_fp_name, 'wb') as handle:
        pickle.dump(mp_fp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    mp_fp = mp_fp[frozenset(get_positions(measurement).items())]
    return mp_fp


def align_images_py(measurement: FolderAlignHistology):
    """
    align_images_py align the images using the points in the mp_fp.mat file

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    """
    # TODO: check if this is correct
    # loads the matched landmark points
    mp_fp_name = os.path.join(measurement.to_align_link, 'mp_fp.pickle')
    mp_fp = {}
    mp_fp_mat = sio.loadmat(mp_fp_name.replace('pickle', 'mat'))
    mp_fp[frozenset(get_positions(measurement).items())] = {'mp': mp_fp_mat['mp'], 'fp': mp_fp_mat['fp']}
    with open(mp_fp_name, 'wb') as handle:
        pickle.dump(mp_fp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    mp_fp = mp_fp[frozenset(get_positions(measurement).items())]
    return mp_fp
    
    
def align_images_imgj(measurement: FolderAlignHistology):
    """
    align_images_imgj align the images using the points in the mp_fp.mat file and imageJ

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    """
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)).split('src')[0], 'notebooks', 'temp')
    histology_aligned = np.array(Image.open(os.path.join(dir_path, 'histo_registered.tif')))
    labels_aligned = np.array(Image.open(os.path.join(dir_path, 'labels_registered.tif')))
    labels_GM_WM_aligned = np.array(Image.open(os.path.join(dir_path, 'labels_GM_WM_registered.tif')))
    [measurement.histology_aligned, measurement.labels_aligned, measurement.labels_GM_WM_aligned] = [histology_aligned, labels_aligned, labels_GM_WM_aligned]


def save_aligned_images(measurement):
    """
    save_aligned_images save the aligned images into the correct folder

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    """
    path_histology_polarimetry_aligned = os.path.join(measurement.path_histology_polarimetry, 'aligned')
    Image.fromarray(measurement.histology_aligned).save(os.path.join(path_histology_polarimetry_aligned, 
                                                                    'histology_aligned_MLS.png'))
    Image.fromarray(measurement.labels_aligned).save(os.path.join(path_histology_polarimetry_aligned, 
                                                                    'labels_aligned_MLS.png'))
    Image.fromarray(measurement.labels_GM_WM_aligned).save(os.path.join(path_histology_polarimetry_aligned, 
                                                                    'labels_GM_WM_aligned_MLS.png'))


def move_the_alignment_results(measurement):
    """
    move_the_alignment_results is used to move the results into the aligned folder

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    """
    path_histology_polarimetry_aligned = os.path.join(measurement.path_histology_polarimetry, 'aligned')
    try:
        os.mkdir(path_histology_polarimetry_aligned)
    except:
        pass
    
    # move the points used for the registration
    # mat_fname_target = os.path.join(measurement.path_histology_polarimetry, 'mp_fp.mat')
    # mat_fname = os.path.join(measurement.to_align_link, 'mp_fp.mat')
    # shutil.copy(mat_fname, mat_fname_target)
    
    # move the points used for the registration
    mat_fname_target = os.path.join(measurement.path_histology_polarimetry, 'mp_fp.pickle')
    mat_fname = os.path.join(measurement.to_align_link, 'mp_fp.pickle')
    shutil.copy(mat_fname, mat_fname_target)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dir_path, 'alignment', 'aligned')
    try:
        os.mkdir(fname)
    except:
        pass
    source_dir = measurement.to_align_link
    target_dir = measurement.to_align_link.replace('to_align', 'aligned').split('/')[0]
    shutil.move(source_dir, target_dir)
    measurement.to_align_link = measurement.to_align_link.replace('to_align', 'aligned')


def generate_final_masks(alignment_measurements, Verbose: bool = False):
    for measurement in tqdm(alignment_measurements):
        get_biggest_blob(measurement)
            

        start = time.time()
        img_labels_propagateds = create_the_image_labels_propagated(measurement)
        measurement.labels_final = img_labels_propagateds[0]
        measurement.labels_GM_WM_final = img_labels_propagateds[1]
        end = time.time()
        if Verbose:
            print("create_the_image_labels_propagated: {:.3f} seconds.".format(end - start))
        
        distance = 0
        measurement.mask_border = get_mask_border(measurement, distance = distance)
        
        start = time.time()
        get_final_mask(measurement)
        end = time.time()
        if Verbose:
            print("get_final_mask: {:.3f} seconds.".format(end - start))
            
        start = time.time()
        save_all_combined_maps(measurement)
        end = time.time()
        if Verbose:
            print("save_all_combined_maps: {:.3f} seconds.".format(end - start))
    
    

def correct_after_imageJ(labels, color_maps):
    finalized = np.zeros(labels.shape)
    
    for type_, colors in color_maps.items():
        if type_ != 'Background':
            label = (np.sum(labels == colors['RGB'], axis = 2) == 3).astype(np.uint8)

            kernel = np.ones((5, 5), np.uint8)
            image = cv2.erode(label, kernel) 
            image = cv2.dilate(image, kernel, cv2.BORDER_REFLECT, iterations = 2) 
            image = morphology.area_closing(image, area_threshold = 64, connectivity=2)

            for idx, idy in zip(np.where(image == 1)[0], np.where(image == 1)[1]):
                finalized[idx, idy] = colors['RGB']
    
    return finalized


def find_nearest_white(nonzero, target, distances_1st_axis, distances_2nd_axis):
    """
    find the nearest pixel to target which is in the nonzero array

    Parameters
    ----------
    nonzero : np.array
        the array containing the indexes of the pixels of interest
    target : tuple
        the target pixel coordinates

    Returns
    -------
    nonzero[nearest_index] : tuple
        the closest pixel to target which is in the nonzero array
    """
    distances = distances_1st_axis[target[1]] + distances_2nd_axis[target[0]]
    distances = distances.reshape(distances.shape[0],)
    all_nearest = np.argpartition(distances,5)[:5]
    return nonzero[all_nearest]

def get_final_mask(measurement, wavelength = '550'):
    path_polarimetry = measurement.folder_path
    path_histology_polarimetry = measurement.path_histology_polarimetry
    MM = np.load(os.path.join(path_polarimetry, 'polarimetry', '550nm', 'MM.npz'))
    mask = MM['Msk']
    mask = np.ones(mask.shape)
    
    img_labels_propagated, img_labels_propagated_GM_WM = measurement.labels_final, measurement.labels_GM_WM_final
    mask_border = measurement.mask_border
    img_labels_propagated_kept = np.zeros(img_labels_propagated.shape)
    for idx, x in enumerate(img_labels_propagated):
        for idy, y in enumerate(x):
            if not mask[idx, idy]:
                pass
            else:
                if mask_border[idx, idy] > 0:
                    img_labels_propagated_kept[idx, idy] = y
    img_labels_propagated = img_labels_propagated_kept.astype(np.uint8)
    
    img_labels_propagated_kept = np.zeros(img_labels_propagated_GM_WM.shape)
    for idx, x in enumerate(img_labels_propagated_GM_WM):
        for idy, y in enumerate(x):
            if not mask[idx, idy]:
                pass
            else:
                if mask_border[idx, idy] > 0:
                    img_labels_propagated_kept[idx, idy] = y
    img_labels_propagated_GM_WM = img_labels_propagated_kept.astype(np.uint8)
    
    Image.fromarray(img_labels_propagated).save(os.path.join(path_histology_polarimetry, 
                                                                 'labels_augmented_masked.png'))
    Image.fromarray(img_labels_propagated_GM_WM).save(os.path.join(path_histology_polarimetry, 
                                                                   'labels_augmented_GM_WM_masked.png'))
    
    overlay_img(measurement.img_polarimetry_gs,
                img_labels_propagated, 
                os.path.join(path_histology_polarimetry, 'overlay_final_masked.png'))
    overlay_img(measurement.img_polarimetry_gs,
                img_labels_propagated_GM_WM, 
                os.path.join(path_histology_polarimetry, 'overlay_final_GM_WM_masked.png'))
    
    measurement.labels_final = img_labels_propagated
    measurement.labels_GM_WM_final = img_labels_propagated_GM_WM

def overlay_img(path_bg, path_fg, save_path):
    if type(path_bg) == str:
        background = Image.open(path_bg)
    elif type(path_bg) == np.ndarray:
        background = Image.fromarray(path_bg)
    else:
        background = path_bg
    if type(path_fg) == str:
        overlay = Image.open(path_fg)
    elif type(path_fg) == np.ndarray:
        overlay = Image.fromarray(path_fg)
    else:
        overlay = path_fg

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    new_img = Image.blend(background, overlay, 0.5)
    new_img.save(save_path,"PNG")
    
    
import random
random.seed(0)

def create_the_image_labels_propagated(measurement, val: int = 50):
    
    color_code_link = load_color_code_links()
    color_code_link_GM_WM = load_color_code_links(GM_WM = True)
    
    color_map = load_color_maps()
    color_map_GM_WM = load_color_maps(GM_WM = True)

    labels = measurement.labels_aligned_blobed
    labels = correct_after_imageJ(labels, color_map)
    labels_GM_WM = measurement.labels_GM_WM_aligned_blobed
    labels_GM_WM = correct_after_imageJ(labels_GM_WM, color_map_GM_WM)
    
    labels_L = np.array(Image.fromarray(measurement.labels_aligned_blobed.astype(np.uint8)).convert('L'))
    labels_GM_WM_L = np.array(Image.fromarray(measurement.labels_GM_WM_aligned_blobed.astype(np.uint8)).convert('L'))
    
    img_labels_propagated = np.zeros(labels.shape)
    img_labels_propagated_GM_WM = np.zeros(labels_GM_WM.shape)

    nonzero = cv2.findNonZero(labels_L)
    nonzero = np.array(random.sample(list(nonzero), round(len(nonzero) / val)))
    
    nonzero_WM_GM = cv2.findNonZero(labels_GM_WM_L)
    try:
        nonzero_WM_GM = np.array(random.sample(list(nonzero_WM_GM), round(len(nonzero_WM_GM) / val)))
        process_WM_GM = True
    except:
        process_WM_GM = False

    # precompute the distances
    distances_1st_axis = {}
    for idx in range(labels_L.shape[1]):
        distances_1st_axis[idx] = (nonzero[:,:,0] - idx) **2
    distances_2nd_axis = {}
    for idx in range(labels_L.shape[1]):
        distances_2nd_axis[idx] = (nonzero[:,:,1] - idx) **2
        
    if process_WM_GM:
        # precompute the distances
        distances_1st_axis_GM_WM = {}
        for idx in range(labels_GM_WM_L.shape[1]):
            distances_1st_axis_GM_WM[idx] = (nonzero_WM_GM[:,:,0] - idx) **2
        distances_2nd_axis_GM_WM = {}
        for idx in range(labels_GM_WM_L.shape[1]):
            distances_2nd_axis_GM_WM[idx] = (nonzero_WM_GM[:,:,1] - idx) **2
    
    ROI = np.array(Image.open(measurement.annotation_path))
    
    for idx, x in enumerate(ROI):
        for idy, y in enumerate(x):
            if y == 0:
                pass
            else:
                new_labeled = None
                # generate the label for the labels
                if sum(labels[idx, idy]) != 0:
                    new_label = labels[idx, idy].astype(np.uint8)
                    new_labeled = [idx, idy]
                else:
                    target = (idx, idy)
                    new_labeled = find_nearest_white(nonzero, target, distances_1st_axis, distances_2nd_axis, distance = 5)
                    new_label = find_new_labels(new_labeled, labels, color_code_link)
                    counter = 0
                    while sum(new_label) == 0:
                        distance = 25 * (counter+1)
                        new_labeled = find_nearest_white(nonzero, target, distances_1st_axis, distances_2nd_axis, distance = distance)
                        new_label = find_new_labels(new_labeled, labels, color_code_link)
                        counter += 1
                        
                img_labels_propagated[idx, idy] = new_label
                    
                if process_WM_GM:
                    if sum(labels_GM_WM[idx, idy]) != 0:
                        new_label = labels_GM_WM[idx, idy].astype(np.uint8)
                    else:
                        target = (idx, idy)
                        new_labeled = find_nearest_white(nonzero_WM_GM, target, distances_1st_axis_GM_WM, distances_2nd_axis_GM_WM, distance = 5)
                        new_label = find_new_labels(new_labeled, labels_GM_WM, color_code_link_GM_WM, WM = True)
                        counter = 0
                        while sum(new_label) == 0:
                            distance = 25 * (counter+1)
                            new_labeled = find_nearest_white(nonzero_WM_GM, target, distances_1st_axis_GM_WM, distances_2nd_axis_GM_WM, distance = distance)
                            new_label = find_new_labels(new_labeled, labels_GM_WM, color_code_link_GM_WM, WM = True)
                            counter += 1
                        
                        target = (idx, idy)
                        
                    
                    img_labels_propagated_GM_WM[idx, idy] = np.array(new_label).astype(np.uint8)
                    
                    """
                    elif type(new_labeled) == list and len(new_labeled) == 2:
                        if sum(labels_GM_WM[new_labeled[0], new_labeled[1]]) != 0:
                            # new_label = labels_GM_WM[new_labeled[0], new_labeled[1]]
                            pass
                    else:
                        if len(new_labeled) == 1:
                            if sum(labels_GM_WM[new_labeled[0][0], new_labeled[0][1]]) != 0:
                                new_label = labels_GM_WM[new_labeled[0][0], new_labeled[0][1]]
                            raise NotImplementedError
                        elif type(new_labeled) != np.array:
                            process_flag = False
                            new_label = find_new_labels(new_labeled, labels_GM_WM, color_code_link_GM_WM, WM = True)
                            if sum(new_label) != 0:
                                pass
                            else:
                                process_flag = True
                            
                            if process_flag:
                                target = (idx, idy)
                                new_labeled = find_nearest_white(nonzero_WM_GM, target, distances_1st_axis_GM_WM, distances_2nd_axis_GM_WM)
                                new_label = find_new_labels(new_labeled, labels_GM_WM, color_code_link_GM_WM, WM = True)
                                raise NotImplementedError
                    """

    img_labels_propagated = img_labels_propagated.astype(np.uint8)
    img_labels_propagated_GM_WM = img_labels_propagated_GM_WM.astype(np.uint8)
    measurement.img_labels_propagated = img_labels_propagated
    measurement.img_labels_propagated_GM_WM = img_labels_propagated_GM_WM
    return img_labels_propagated, img_labels_propagated_GM_WM


def find_new_labels(new_labeled, labels, color_code_link, WM = False):
    all_labels = []
        
    for label in new_labeled:
        idx_min, idy_min = label[0][1], label[0][0]
            
        if not WM:
            labeled = labels[idx_min, idy_min].astype(np.uint8)
        else:
            labeled = labels[idx_min, idy_min].astype(np.uint8)

        new_label = None
        for val in color_code_link.keys():
            tot_distance = 0
            for l, v in zip(labeled, val):
                tot_distance += abs(l - v)
            if tot_distance == 0:
                new_label = list(val)


        if new_label == None:
            distances = []
            for val in color_code_link.keys():
                tot_distance = 0
                for l, v in zip(labeled, val):
                    tot_distance += abs(l - v)
                distances.append(tot_distance)
            new_label = list(list(color_code_link.keys())[np.argmin(distances)])

        all_labels.append(tuple(new_label))
        
    return list(most_frequent(all_labels))


def mask_array(image, mask):
    masked_img = np.zeros(image.shape)
    for idx, x in enumerate(mask):
        for idy, y in enumerate(x):
            masked_img[idx, idy] = image[idx, idy] * y
    return masked_img

def get_biggest_blob(measurement):
    img_bw = np.sum(measurement.labels_aligned, axis = 2) != 0
    kernel = np.ones((5, 5), np.uint8)
    img_bw = cv2.erode(img_bw.astype(np.uint8), kernel, iterations=1)
    labels = skimage.measure.label(img_bw, return_num=False)
    maxCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=img_bw.flat))
    measurement.labels_aligned_blobed = mask_array(measurement.labels_aligned, maxCC_nobcg).astype(np.uint8)
    measurement.labels_GM_WM_aligned_blobed = mask_array(measurement.labels_GM_WM_aligned, maxCC_nobcg).astype(np.uint8)
    
    


def find_nearest_white(nonzero, target, distances_1st_axis, distances_2nd_axis, bd = False, distance = 5):
    """
    find the nearest pixel to target which is in the nonzero array

    Parameters
    ----------
    nonzero : np.array
        the array containing the indexes of the pixels of interest
    target : tuple
        the target pixel coordinates

    Returns
    -------
    nonzero[nearest_index] : tuple
        the closest pixel to target which is in the nonzero array
    """
    distances = distances_1st_axis[target[1]] + distances_2nd_axis[target[0]]
    if not bd:
        distances = distances.reshape(distances.shape[0],)
        all_nearest = np.argpartition(distances,distance)[:distance]
        return nonzero[all_nearest]
    elif bd:
        nearest_index = np.argmin(distances)
        return nonzero[nearest_index]



def save_all_combined_maps(measurement):
    
    make_overlays_maps(measurement)
    
    figures = ['Depolarization_img 550nm.png', 'Azimuth of optical axis_img 550nm.png', 
               'Intensity_img 550nm.png', 'Linear retardance_img 550nm.png']
    name = 'combined_550.png'
    save_combined_img(figures, name, os.path.join(measurement.path_histology_polarimetry, 'maps_aligned'))
    
    figures = ['Depolarization_img 550nmGM_WM.png', 'Azimuth of optical axis_img 550nmGM_WM.png', 
               'Intensity_img 550nmGM_WM.png', 'Linear retardance_img 550nmGM_WM.png']
    name = 'combined_550_GM_WM.png'
    save_combined_img(figures, name, os.path.join(measurement.path_histology_polarimetry, 'maps_aligned'))
    
                        
def save_combined_img(figures, name, path_histology_polarimetry):
    path = path_histology_polarimetry
    
    titles = ['Depolarization', 'Azimuth of optical axis (°)', 'Intensity', 'Linear retardance']

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,7))
    for idx, (fig, title) in enumerate(zip(figures, titles)):
        row = idx%2
        col = idx//2
        ax = axes[row, col]
        ax.axis('off')
        img = plt.imread(os.path.join(path, fig))
        ax.imshow(img)
        ax.set_title(title, fontsize="20", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(path, name))
    plt.close()
    
    
def make_overlays_maps(measurement):
    """
    creates an overlay for each of the polarimetric maps

    Parameters
    ----------
    path_histology_polarimetry : str
        the path to the folder in the to_align subfolder
    img_labels_propagated_RGB : Pillow Image
        the final labels image in the RGB format
    final_mask : boolean
        indicates if we are working with the mask with the border removed (default = False)
    """
    figures = ['Depolarization_img.png', 'Azimuth of optical axis_img.png', 
               'Intensity_img.png', 'Linear retardance_img.png']
    
    try:
        os.mkdir(os.path.join(measurement.path_histology_polarimetry, 'maps_aligned'))
    except:
         pass
    path_polarimetry = os.path.join(measurement.folder_path, 'polarimetry')
    
    for wavelength in os.listdir(path_polarimetry):
        if '550nm' in wavelength:
            path_polarimetry_wl = os.path.join(path_polarimetry, wavelength)
            for file in os.listdir(path_polarimetry_wl):
                if file in figures:
                    
                    background = Image.open(os.path.join(path_polarimetry_wl, file))
                    overlay = measurement.labels_final
                    fname = os.path.join(measurement.path_histology_polarimetry, 'maps_aligned', 
                                            file.replace('.png', ' ' + wavelength + '.png'))
                    
                    if 'Intensity' in file:
                        make_overlay(background, overlay, fname, alpha = 0.25)
                    else:
                        make_overlay(background, overlay, fname, alpha = 0.6)
                        
                    overlay = measurement.labels_GM_WM_final

                    fname = os.path.join(measurement.path_histology_polarimetry, 'maps_aligned', 
                                    file.replace('.png', ' ' + wavelength + 'GM_WM.png'))
                    if 'Intensity' in file:
                        make_overlay(background, overlay, fname, alpha = 0.25)
                    else:
                        make_overlay(background, overlay, fname, alpha = 0.6)
                        
def make_overlay(background, overlay, fname, alpha = 0.1):
    """
    creates an overlay of background and overlay image

    Parameters
    ----------
    background : Pillow image
        the image to be used as background
    overlay : Pillow image
        the image to overlay on the background
    fname : str
        the path in which the image should be saved
    pol_maps : boolean
        indication of if we are working with polarimetric maps (in this case, crop the image - default = False)
    alpha : double
        the transparency level (default = 0.1)
    """
    background = background.convert('RGBA')
    background.save('img.png')
    if type(overlay) == np.ndarray:
        overlay = Image.fromarray(overlay)
    else:
        pass
    overlay = overlay.convert('RGBA')
    new_img = Image.blend(background, overlay, alpha)
    new_img.save(fname)
       
           




def get_mask_border(measurement, distance = 10):
    """
    is used to remove the border of the image (i.e. remove the possible border effect for the sample)

    Parameters
    ----------
    img_labels_propagated : Pillow Image
        the final labels image
    distance : int
        the distance to remove (default = 30)

    Returns
    -------
    mask_border : np.array
        an array containing the pixels to keep and to remove
    """
    img = (np.sum(measurement.labels_final, axis = 2) != 0).astype(np.uint8)

    # Creating kernel
    kernel = np.ones((distance, distance), np.uint8)
    
    # Using cv2.erode() method 
    image = cv2.erode(img, kernel) 
                
    return image


def write_mp_fp(measurement):
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)).split('src')[0], 'notebooks', 'temp')
    # write mp_fp
    mp_fp = get_mp_fp(measurement)
    text = 'Index\txSource\tySource\txTarget\tyTarget\n'
    for index, (idx, idy) in enumerate(zip(mp_fp['fp'], mp_fp['mp'])):
        line = str(index) + '\t' + str(round(idx[0])) + '\t' + str(round(idx[1])) + '\t' + str(round(idy[0])) + '\t' + str(round(idy[1])) +'\n' 
        text += line
    with open(os.path.join(dir_path, 'coordinates.txt'), 'w') as f:
        f.write(text)
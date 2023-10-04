import time
from tqdm import tqdm
import numpy as np
import cv2
import skimage
from PIL import Image
import random
random.seed(0)

from histopolalign.prepare_images import FolderAlignHistology
from histopolalign.helpers import load_color_code_links, most_frequent, load_color_maps


def generate_final_masks(alignment_measurements: list, Verbose: bool = False):
    """
    generate_final_masks is the master function allowing to propagate the labels in the foreground region of the polarimetry images without any labels (see the paper, section Methods : Mapping pathology and polarimetry, for more details)

    Parameters
    ----------
    alignment_measurements : list of FolderAlignHistology
        the list of the different FolderAlignHistology objects containing the information about the different measurements
    Verbose : bool
        whether to print the time taken by the different steps (default is False)

    Returns
    -------
    nonzero[nearest_index] : tuple
        the closest pixel to target which is in the nonzero array
    """
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
    
    
def get_biggest_blob(measurement):
    """
    get_biggest_blob is the function allowing to get the biggest blob in the histology images

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology object containing the information about the different measurements

    Returns
    -------
    """
    # get a binary image indicating the presence of labels
    img_bw = np.sum(measurement.labels_aligned, axis = 2) != 0
    
    # perform erosion step to remove the small labels
    kernel = np.ones((5, 5), np.uint8)
    img_bw = cv2.erode(img_bw.astype(np.uint8), kernel, iterations=1)
    labels = skimage.measure.label(img_bw, return_num=False)
    
    # get the biggest blob
    maxCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=img_bw.flat))
    
    measurement.labels_aligned_blobed = mask_array(measurement.labels_aligned, maxCC_nobcg).astype(np.uint8)
    measurement.labels_GM_WM_aligned_blobed = mask_array(measurement.labels_GM_WM_aligned, maxCC_nobcg).astype(np.uint8)
    

def mask_array(image: np.array, mask: np.array):
    """
    mask_array is the function allowing to mask an image with a binary mask

    Parameters
    ----------
    image : np.array
        the image to be masked
    mask : np.array
        the binary mask

    Returns
    -------
    masked_img : np.array
        the masked image
    """
    masked_img = np.zeros(image.shape)
    for idx, x in enumerate(mask):
        for idy, y in enumerate(x):
            masked_img[idx, idy] = image[idx, idy] * y
    return masked_img


def create_the_image_labels_propagated(measurement: FolderAlignHistology, val: int = 50):
    """
    create_the_image_labels_propagated is the function creating the final labels, expanding the labels in the foreground region of the polarimetry images without any labels (see the paper, section Methods : Mapping pathology and polarimetry, for more details)

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology object containing the information about the different measurements
    val : int
        the value to be used to subsample the image (speeds up the process, default is 50)

    Returns
    -------
    img_labels_propagated, img_labels_propagated_GM_WM : np.array, np.array
        the final labels, expanding the labels in the foreground region of the polarimetry images without any labels
    """
    # load the link between the RGB colors and the int code
    color_code_link = load_color_code_links()
    color_code_link_GM_WM = load_color_code_links(GM_WM = True)
    
    # load the parameters for the histogram plots
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

    img_labels_propagated = img_labels_propagated.astype(np.uint8)
    img_labels_propagated_GM_WM = img_labels_propagated_GM_WM.astype(np.uint8)
    measurement.img_labels_propagated = img_labels_propagated
    measurement.img_labels_propagated_GM_WM = img_labels_propagated_GM_WM
    return img_labels_propagated, img_labels_propagated_GM_WM


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


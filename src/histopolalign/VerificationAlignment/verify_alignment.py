import numpy as np
from PIL import Image, ImageFilter
import cv2
from tqdm import tqdm
import os

def compute_similarity(data_folder: list):
    """
    compute_similarity is the master function to compute the dice scores for the alignments and save the overimposed images

    Parameters
    ----------
    data_folder : list
        the list of the folders to be considered
        
    Returns
    ----------
    dice_scores : dict
        the dice scores for the alignments
    """
    dice_scores = {}
    size_neighbors = 4

    for folder in tqdm(data_folder):
        image_gt, image_labels_raw, image_labels_inti, image_labels_contour, image_labels_aligned = load_labels(folder)
        border_gt, border_raw, border_inti, border_contour, border_aligned = get_borders([image_gt, image_labels_raw, image_labels_inti, image_labels_contour, image_labels_aligned], folder, size_neighbors)

        img_reference = border_gt
        img_aligned = [border_raw, border_inti, border_contour, border_aligned]

        # initialize the dice scores
        dice_score = {'raw': 0, 'pre-align': 0, 'contour': 0, 'MLS': 0}
        
        # compute the dice scores for each steo of the alignment and save the overimposed images
        for img, (key,_) in zip(img_aligned, dice_score.items()):
            dice_score[key] = dice(img_reference, img, k = 1)
            overimposed = border_gt + img
            overimposed = Image.fromarray(overimposed.astype(np.uint8) * 120)
            overimposed.save(os.path.join(folder, 'histology', 'overimposed_' + key + '.png'))
            
        dice_scores[folder] = dice_score
        
    return dice_scores


def load_labels(folder: str):
    """
    load the labels after each of the steps of the alignment

    Parameters
    ----------
    folder : str
        the path to the folder containing the annotations considered
        
    Returns
    ----------
    image_gt, image_labels_raw, image_labels_inti, image_labels_contour, image_labels_aligned : np.array
        the images of the labels after each of the steps of the alignment
    """
    image_gt_path = os.path.join(folder, 'annotation', 'merged.png')
    image_gt = np.array(Image.open(image_gt_path))
                
    image_labels_raw_path = os.path.join(folder, 'histology', 'labels_GM_WM_original.png')
    image_labels_raw = np.array(Image.open(image_labels_raw_path))
        
    image_labels_init_path = os.path.join(folder, 'histology', 'labels_GM_WM.png')
    image_labels_inti = np.array(Image.open(image_labels_init_path))
        
    image_labels_contour_path = os.path.join(folder, 'histology', 'aligned' ,'histology_labels_GM_WM_contour.png')
    image_labels_contour = np.array(Image.open(image_labels_contour_path))
        
    image_labels_aligned_path = os.path.join(folder, 'histology', 'aligned' ,'labels_GM_WM_aligned_MLS.png')
    image_labels_aligned = np.array(Image.open(image_labels_aligned_path))
    image_labels_aligned = create_image_labels_final(image_labels_aligned, colors = [[153, 153, 0], [153, 77, 0]])
    
    return image_gt, image_labels_raw, image_labels_inti, image_labels_contour, image_labels_aligned

    
def create_image_labels_final(image_labels: np.array, colors = []):
    """
    create_image_labels_final is the function to create the dilatted image labels

    Parameters
    ----------
    image_labels : np.array
        the image of the labels
    colors : list
        the colors of the labels in the current image
        
    Returns
    ----------
    img_final : np.array
        the image of the labels after the dilation
    """
    # initialize the final image
    img_final = np.zeros(image_labels.shape)
    
    # iterate over the colors and dilate them slightly
    for color in colors:
        img = (np.sum(image_labels == color, axis = 2) == 3).astype(np.uint8)
        img = Image.fromarray(img * 255)
        img = np.array(img.filter(ImageFilter.ModeFilter(size=15)))
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=3)
        
        for idx, x in enumerate(img):
            for idy, y in enumerate(x):
                if y != 0:
                    img_final[idx, idy] = color
    
    return img_final.astype(np.uint8)


def get_borders(imgs: list, folder: str, size_neighbors: int = 4):
    """
    get_borders is the function to get the borders for the labels at different steps of the alignment

    Parameters
    ----------
    imgs : list
        the list of the images of the labels after each of the steps of the alignment
    folder : str
        the path to the folder containing the annotations considered
    size_neighbors : int
        the size of the neighbors to consider to define the borders
        
    Returns
    ----------
    border_gt_img, border_raw_img, border_inti_img, border_contour_img, border_aligned_img : np.array
        the borders for the labels at different steps of the alignment
    """
    image_gt, image_labels_raw, image_labels_inti, image_labels_contour, image_labels_aligned = imgs
    border_gt = get_border(image_gt, img_labels = False, size_neighbors = size_neighbors, colors = [128, 255])
    Image.fromarray(border_gt.astype(np.uint8) * 255).save(os.path.join(folder, 'histology', 'border_ground_truth.png'))

    border_raw = get_border(image_labels_raw, img_labels = True, size_neighbors = size_neighbors, colors = [[153, 153, 0], [153, 77, 0]])
    Image.fromarray(border_raw.astype(np.uint8) * 255).save(os.path.join(folder, 'histology', 'border_raw.png'))
        
    border_inti = get_border(image_labels_inti, img_labels = True, size_neighbors = size_neighbors, colors = [[153, 153, 0], [153, 77, 0]])
    Image.fromarray(border_inti.astype(np.uint8) * 255).save(os.path.join(folder, 'histology', 'border_initialized.png'))
       
    border_contour = get_border(image_labels_contour, img_labels = True, size_neighbors = size_neighbors, colors = [[153, 153, 0], [153, 77, 0]])
    Image.fromarray(border_contour.astype(np.uint8) * 255).save(os.path.join(folder, 'histology', 'border_contour.png'))
        
    border_aligned = get_border(image_labels_aligned, img_labels = True, size_neighbors = size_neighbors, colors = [[153, 153, 0], [153, 77, 0]])
    Image.fromarray(border_aligned.astype(np.uint8) * 255).save(os.path.join(folder, 'histology', 'border_aligned.png'))
    
    return border_gt, border_raw, border_inti, border_contour, border_aligned
        

def get_border(img: np.array, img_labels: bool = False, size_neighbors: int = 4, colors: list = [[153, 153, 0], [153, 77, 0]]):
    """
    get_border is the function to get the border between the WM and the GM for one label

    Parameters
    ----------
    img : np.array
        the image of the labels
    img_labels : bool
        whether the image is a label image or ground truth
    size_neighbors : int
        the size of the neighbors to consider to define the borders
    colors : list
        the colors of the labels in the current image
        
    Returns
    ----------
    border : np.array
        the image of the borders
    """
    border = np.zeros(img.shape[0:2])

    # iterate over the pixels of the image
    for idx, x in enumerate(img):
        for idy, _ in enumerate(x):
            if img_labels:
                condition = sum(img[idx, idy]) != 0
            else:
                condition = img[idx, idy] != 0
            
            # get the neighbors of the pixel
            if condition:
                neighbors = img[idx - size_neighbors+1:idx + size_neighbors, idy - size_neighbors + 1:idy + size_neighbors]
                
                try:
                    neighbors = neighbors.reshape(-1, neighbors.shape[-1])

                    # check if the pixel is a border pixel
                    for color in colors:
                        if img_labels:
                            if tuple(img[idx, idy]) == tuple(color):
                                for col in colors:
                                    is_border = False
                                    if col == color:
                                        pass
                                    else:
                                        if np.sum(np.sum(tuple(col) == np.unique(neighbors, axis=0), axis = 1) == 3) != 0:
                                            is_border = True
                        else:
                            if img[idx, idy] == color:
                                for col in colors:
                                    is_border = False
                                    if col == color:
                                        pass
                                    else:
                                        if np.sum(np.sum(col == np.unique(neighbors, axis=0), axis = 1) != 0) != 0:
                                            is_border = True
                    
                    if is_border:
                        border[idx, idy] = 1
                except:
                    pass
    
    # dilate the borders to reach 1.6mm, the size of the uncertainty zone (R. Gros et al, Neurophotonics, 2023)
    kernel = np.ones((6, 6), np.uint8)
    border = cv2.dilate(border, kernel, iterations=5)

    return border


def dice(pred: np.array, true: np.array, k: int = 1):
    """
    dice computes the dice scores between two images

    Parameters
    ----------
    pred, true : np.array
        the images to compare
    k : int
        the label to consider
        
    Returns
    ----------
    dice : float
        the dice score between the two images
    """
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice
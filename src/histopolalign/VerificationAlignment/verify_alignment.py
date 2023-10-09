import numpy as np
from PIL import Image, ImageFilter
import cv2
from tqdm import tqdm
import os

def create_image_labels_final(image_labels_final, colors = []):
    img_final = np.zeros(image_labels_final.shape)
    
    for color in colors:
        img = (np.sum(image_labels_final == color, axis = 2) == 3).astype(np.uint8)
        img = Image.fromarray(img * 255)
        img = np.array(img.filter(ImageFilter.ModeFilter(size=15)))
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=3)
        
        for idx, x in enumerate(img):
            for idy, y in enumerate(x):
                if y != 0:
                    img_final[idx, idy] = color
    
    return img_final.astype(np.uint8)


def get_border(img, img_labels = False, size_neighbors = 4, colors = [[153, 153, 0], [153, 77, 0]]):
    border = np.zeros(img.shape[0:2])

    for idx, x in enumerate(img):
        for idy, y in enumerate(x):
            if img_labels:
                condition = sum(img[idx, idy]) != 0
            else:
                condition = img[idx, idy] != 0
                
            if condition:
                neighbors = img[idx - size_neighbors+1:idx + size_neighbors, idy - size_neighbors + 1:idy + size_neighbors]
                
                try:
                    neighbors = neighbors.reshape(-1, neighbors.shape[-1])

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
                            
    kernel = np.ones((6, 6), np.uint8)
    border = cv2.dilate(border, kernel, iterations=5)

    return border


# Dice similarity function
def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice



def compute_similarity(data_folder):
    dice_scores = {}

    size_neighbors = 4

    for folder in tqdm(data_folder):
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
        
        
        dice_score = {'raw': 0, 'pre-align': 0, 'contour': 0, 'MLS': 0}
        
        border_gt = get_border(image_gt, img_labels = False, size_neighbors = size_neighbors, colors = [128, 255])
        border_gt_img = Image.fromarray(border_gt.astype(np.uint8) * 255)
        border_gt_img.save(os.path.join(folder, 'histology', 'border_ground_truth.png'))
        
        border_raw = get_border(image_labels_raw, img_labels = True, size_neighbors = size_neighbors, colors = [[153, 153, 0], [153, 77, 0]])
        border_raw_img = Image.fromarray(border_raw.astype(np.uint8) * 255)
        border_raw_img.save(os.path.join(folder, 'histology', 'border_raw.png'))
        
        border_inti = get_border(image_labels_inti, img_labels = True, size_neighbors = size_neighbors, colors = [[153, 153, 0], [153, 77, 0]])
        border_inti_img = Image.fromarray(border_inti.astype(np.uint8) * 255)
        border_inti_img.save(os.path.join(folder, 'histology', 'border_initialized.png'))
        
        border_contour = get_border(image_labels_contour, img_labels = True, size_neighbors = size_neighbors, colors = [[153, 153, 0], [153, 77, 0]])
        border_contour_img = Image.fromarray(border_contour.astype(np.uint8) * 255)
        border_contour_img.save(os.path.join(folder, 'histology', 'border_contour.png'))
        
        border_aligned = get_border(image_labels_aligned, img_labels = True, size_neighbors = size_neighbors, colors = [[153, 153, 0], [153, 77, 0]])
        border_aligned_img = Image.fromarray(border_aligned.astype(np.uint8) * 255)
        border_aligned_img.save(os.path.join(folder, 'histology', 'border_aligned.png'))
        
        img_reference = border_gt
        img_aligned = [border_raw, border_inti, border_contour, border_aligned]

        for img, (key,_) in zip(img_aligned, dice_score.items()):
            dice_score[key] = dice(img_reference, img, k = 1)
            overimposed = border_gt + img
            overimposed = Image.fromarray(overimposed.astype(np.uint8) * 120)
            overimposed.save(os.path.join(folder, 'histology', 'overimposed_' + key + '.png'))
            
        dice_scores[folder] = dice_score
        
    return dice_scores
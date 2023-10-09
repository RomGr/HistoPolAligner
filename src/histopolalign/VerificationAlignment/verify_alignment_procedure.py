import os
import numpy as np
from histopolalign.VerificationAlignment import mask_generation, verify_alignment
from collections import defaultdict

def verify_alignment_main():
    neoplastic_polarimetry_path = os.path.join(os.getcwd().split('notebooks')[0], 'data', 'TumorMeasurements')
    folder_of_interests = mask_generation.create_the_masks(path_fixation_folder = neoplastic_polarimetry_path)
    dice_scores = verify_alignment.compute_similarity(folder_of_interests)
    
    dice_avg = defaultdict(list)
    for _, val in dice_scores.items():
        for key, dic in val.items():
            dice_avg[key].append(dic)
    dice_mean = {}
    dice_std = {}

    for key, val in dice_avg.items():
        dice_mean[key] = np.mean(val)
        dice_std[key] = np.std(val)
    
    return dice_mean, dice_std
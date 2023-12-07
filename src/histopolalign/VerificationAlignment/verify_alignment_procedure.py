import os
import numpy as np
from histopolalign.VerificationAlignment import mask_generation, verify_alignment
from collections import defaultdict

def verify_alignment_main():
    """
    verify_alignment_main is the main function to verify the images alignment protocol accuracy

    Parameters
    ----------

    Returns
    -------
    dice_mean, dice_std : dict, dict
        the mean and std of the dice scores for the different alignments
    """
    # get the path to the measurements
    neoplastic_polarimetry_path = os.path.join(os.getcwd().split('notebooks')[0], 'data', 'TumorMeasurements')
    folder_of_interests = mask_generation.create_the_masks(path_fixation_folder = neoplastic_polarimetry_path)
    
    # compute the dice scores for the alignments
    dice_scores, assd_scores, hausdorff_distances = verify_alignment.compute_similarity(folder_of_interests)
    
    dice_mean, dice_std = get_mean_std(dice_scores)
    assd_mean, assd_std = get_mean_std(assd_scores)
    hausdorff_mean, hausdorff_std = get_mean_std(hausdorff_distances)
    
    return [dice_mean, dice_std], [assd_mean, assd_std], [hausdorff_mean, hausdorff_std]


def get_mean_std(scores: dict):
    """
    get the mean and std of the scores for the different alignments when working with different images

    Parameters
    ----------
    scores : dict
        the scores for the different alignments

    Returns
    -------
    score_mean, score_std : dict, dict
        the mean and std of the scores for the different alignments
    """
    avg = defaultdict(list)
    for _, val in scores.items():
        for key, dic in val.items():
            avg[key].append(dic)
    score_mean = {}
    score_std = {}

    for key, val in avg.items():
        score_mean[key] = np.mean(val)
        score_std[key] = np.std(val)
        
    return score_mean, score_std
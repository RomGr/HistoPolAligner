import os
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt
from histopolalign.VerificationAlignment import mask_generation, verify_alignment

def verify_alignment_main():
    neoplastic_polarimetry_path = os.path.join(os.getcwd().split('notebooks')[0], 'data', 'TumorMeasurements')
    folder_of_interests = mask_generation.create_the_masks(path_fixation_folder = neoplastic_polarimetry_path)
    verify_alignment.compute_similarity(folder_of_interests)
from tqdm import tqdm
import os
from processingmm import batch_processing
from histopolalign.AlignImages import combine_images, prepare_images, semi_automatic_tool, align_folders, match_skeletons, align_imgs, finalize_mask, get_numerical_results

def align_measurements():
    """
    master function calling the function to align the images for each measurement folder

    Parameters
    ----------
    
    Returns
    -------
    """
    # get the path to the folder containing the measurements
    neoplastic_polarimetry_path = os.path.join(os.getcwd().split('notebooks')[0], 'data', 'TumorMeasurements')
    
    # iterate over the folders containing the measurements and align the images
    for folder in tqdm(os.listdir(neoplastic_polarimetry_path)):
        
        # check that the folder contains measurements
        assert len(os.listdir(os.path.join(neoplastic_polarimetry_path, folder))) > 0, 'No measurements in folder {}'.format(folder)
        align_single_measurement(folder, neoplastic_polarimetry_path = neoplastic_polarimetry_path)
    

def align_single_measurement(neoplastic_folder: str, neoplastic_polarimetry_path: str = None):
    """
    function aligning the measurement for one patient

    Parameters
    ----------
    neoplastic_folder : str
        the name of the folder containing the measurements
    neoplastic_polarimetry_path : str, optional
        the path to the folder containing the measurements, by default None
    
    Returns
    -------
    """
    # 1. Process the measurements that will be aligned
    
    # get the folder in which the polarimetric measurements are stored
    if neoplastic_polarimetry_path is None:
        neoplastic_polarimetry_path = os.path.join(os.getcwd().split('notebooks')[0], 'data', 'TumorMeasurements')
        
    polarimetry_path = os.path.join(neoplastic_polarimetry_path, neoplastic_folder)
    directories = [os.path.join(neoplastic_polarimetry_path, neoplastic_folder)]
    
    # process the measurements, if needed
    calib_directory = os.path.join(os.getcwd().split('notebooks')[0], 'calib')
    batch_processing.batch_process(directories, calib_directory, run_all = False)
    
    
    # 2. Get the combined masks for the histology folders
    
    # create the folde that will be used to align pathology and polarimetry
    combine_images.create_to_align_folder()

    # load the pathology images...
    histology_path = os.path.join(os.getcwd().split('notebooks')[0], 'data', 'HistologyResults')
    imgs_all = combine_images.load_the_images(histology_path, Verbose = False)

    # ...and process them
    combine_images.get_combined_img(imgs_all, force_recompute = False, Verbose = False)
    
    
    # 3. Obtain the parameters (manually using the GUI) to first align the histology and the polarimetry
    
    # Prepare the images
    alignment_measurements = prepare_images.create_the_alignments(histology_path, polarimetry_path, Verbose = False)
    
    # Load the polarimetry, labels and histology (H&E) images for each polarimetry folder
    alignment_measurements = prepare_images.load_and_preprocess_imgs(alignment_measurements, force_recompute = False, Verbose = False)
    
    # Semi-automatic rotation/alignement tool
    for measurement in alignment_measurements:
        _ = semi_automatic_tool.ask_for_parameters(measurement, force_recompute = False)
        
        
    # 4. Actually perform the alignment
    
    align_folders.create_align_folders(alignment_measurements, Verbose = False)
    
    # Automatic part
    border_parameter = 5
    nsamples = 400
    max_distance = 150

    for measurement in alignment_measurements:
        match_skeletons.match_skeletons(measurement, border_parameter, nsamples = nsamples, max_distance = max_distance,
                                                                    Verbose = False)
        
    # Semi-automatic part
    align_folders.semi_automatic_processing(alignment_measurements)
    alignment_measurements = align_imgs.align_img_master(alignment_measurements)
    
    # Get the final masks and overlay them on the polarimetric parameter maps
    finalize_mask.generate_final_masks(alignment_measurements, Verbose = False)
    
    # Finalize and save the numerical results
    _ = get_numerical_results.create_numerical_values(alignment_measurements)
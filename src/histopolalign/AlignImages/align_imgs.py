import numpy as np
from PIL import Image
import os, shutil
import pickle
import scipy.io as sio

from histopolalign.AlignImages.prepare_images import FolderAlignHistology
from histopolalign.AlignImages.align_folders import get_positions


def align_img_master(alignment_measurements: list):
    """
    align_img_master is the master function organizing the data for the alignment and calling the fcuntion actually performing the alignment (in "imgJ_align.py")

    Parameters
    ----------
    alignment_measurements : list of FolderAlignHistology
        the dictionary containing the FolderAlignHistology objects
    
    Returns
    -------
    alignment_measurements : list of FolderAlignHistology
        the dictionary containing the FolderAlignHistology objects with the aligned images
    """
    # get the path to the temp folder used to store the data necessary to align the images
    dir_path = os.path.dirname(os.path.realpath(__file__)).split('src')[0]
    path_temp = os.path.join(dir_path, 'notebooks', 'temp')
    
    # create the temp folder
    try:
        shutil.rmtree(path_temp)
    except FileNotFoundError:
        pass
    try:
        os.mkdir(path_temp)
    except FileExistsError:
        pass
    
    # put the alignment measurements object in the temp folder
    with open(os.path.join(path_temp, 'align_measurement.pickle'), 'wb') as handle:
        pickle.dump(alignment_measurements, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # call the function aligning the images
    os.system("python " + os.path.join(dir_path, 'src', 'histopolalign', 'imgJ_align.py'))
    
    # recover the alignment measurements object from the temp folder
    with open(os.path.join(path_temp, 'align_measurement.pickle'), 'rb') as handle:
        alignment_measurements = pickle.load(handle)
    
    # try to delete the temp folder
    try:
        shutil.rmtree(path_temp)
        pass
    except FileNotFoundError:
        pass
    
    return alignment_measurements
    
    
def align_w_imageJ(ij, alignment_measurements: list):
    """
    align_w_imageJ is the function containing the macro that will be used in ImageJ to align the images

    Parameters
    ----------
    ij :
        the ImageJ session
    alignment_measurements : list
        the list containing the FolderAlignHistology objects to align
    
    Returns
    -------
    """
    # get the current working directory for inserting correctly the file paths in the macro
    current_folder = '"' + os.path.dirname(os.path.realpath(__file__)).split('src')[0].replace('\\', '/')
    current_folder_no_start = current_folder.replace('"', '')
    
    # the macro used to align the images
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
    
    # iterate over the different measurements to align
    for measurement in alignment_measurements:
        write_mp_fp(measurement)
        save_imgs_alignment(measurement)
        ij.py.run_macro(macro)
        recover_aligned_images(measurement)
        save_aligned_images(measurement)
        move_the_alignment_results(measurement)


def write_mp_fp(measurement: FolderAlignHistology):
    """
    write_mp_fp is used to write the coordinates text file used by ImageJ to align the images from the matched points selected in MATLAB

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    
    Returns
    -------
    """
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)).split('src')[0], 'notebooks', 'temp')
    
    # get the matched points
    mp_fp = get_mp_fp(measurement)
    
    # write the header
    text = 'Index\txSource\tySource\txTarget\tyTarget\n'
    
    # write each index, that must be splitted by a tab
    for index, (idx, idy) in enumerate(zip(mp_fp['fp'], mp_fp['mp'])):
        line = str(index) + '\t' + str(round(idx[0])) + '\t' + str(round(idx[1])) + '\t' + str(round(idy[0])) + '\t' + str(round(idy[1])) +'\n' 
        text += line
        
    # write the coordinates text file
    with open(os.path.join(dir_path, 'coordinates.txt'), 'w') as f:
        f.write(text)


def get_mp_fp(measurement: FolderAlignHistology):
    """
    get_mp_fp gets the matched points from a .mat format into a python dictionnary format

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    
    Returns
    -------
    mp_fp : dict
        the python dictionnary containing the matched points
    """    
    mp_fp_name = os.path.join(measurement.to_align_link, 'mp_fp.pickle')
    mp_fp = {}
    
    # load the .mat file containing the matched points
    mp_fp_mat = sio.loadmat(mp_fp_name.replace('pickle', 'mat'))
    
    # save the points in a python dictionary format
    mp_fp[frozenset(get_positions(measurement).items())] = {'mp': mp_fp_mat['mp'], 'fp': mp_fp_mat['fp']}
    with open(mp_fp_name, 'wb') as handle:
        pickle.dump(mp_fp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    mp_fp = mp_fp[frozenset(get_positions(measurement).items())]
    return mp_fp


def save_imgs_alignment(measurement: FolderAlignHistology):
    """
    save_imgs_alignment is used to save the images necessary for ImageJ to align the images in the temp folder

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement

    """    
    # create the path to the temp folder
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
    path_img = os.path.join(dir_path, 'notebooks', 'temp', 'polarimetry.png')
    measurement.img_polarimetry_gs_650.save(path_img)
    

def align_images_py(measurement: FolderAlignHistology):
    """
    align_images_py align the images using the points in the mp_fp.mat file

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    """
    # loads the matched landmark points
    mp_fp_name = os.path.join(measurement.to_align_link, 'mp_fp.pickle')
    mp_fp = {}
    mp_fp_mat = sio.loadmat(mp_fp_name.replace('pickle', 'mat'))
    mp_fp[frozenset(get_positions(measurement).items())] = {'mp': mp_fp_mat['mp'], 'fp': mp_fp_mat['fp']}
    with open(mp_fp_name, 'wb') as handle:
        pickle.dump(mp_fp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    mp_fp = mp_fp[frozenset(get_positions(measurement).items())]
    return mp_fp
    
    
def recover_aligned_images(measurement: FolderAlignHistology):
    """
    recover_aligned_images load the aligned images and save them in the FolderAlignHistology object

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    """
    # load the aligned images
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)).split('src')[0], 'notebooks', 'temp')
    histology_aligned = np.array(Image.open(os.path.join(dir_path, 'histo_registered.tif')))
    labels_aligned = np.array(Image.open(os.path.join(dir_path, 'labels_registered.tif')))
    labels_GM_WM_aligned = np.array(Image.open(os.path.join(dir_path, 'labels_GM_WM_registered.tif')))
    
    # and save them in the FolderAlignHistology object
    [measurement.histology_aligned, measurement.labels_aligned, measurement.labels_GM_WM_aligned] = [histology_aligned, labels_aligned, labels_GM_WM_aligned]


def save_aligned_images(measurement: FolderAlignHistology):
    """
    save_aligned_images save the aligned images into the correct folder

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    """
    # get the path to the folder containing the polarimetric measurements
    path_histology_polarimetry_aligned = os.path.join(measurement.path_histology_polarimetry, 'aligned')
    
    # and save the aligned images
    Image.fromarray(measurement.histology_aligned).save(os.path.join(path_histology_polarimetry_aligned, 
                                                                    'histology_aligned_MLS.png'))
    Image.fromarray(measurement.labels_aligned).save(os.path.join(path_histology_polarimetry_aligned, 
                                                                    'labels_aligned_MLS.png'))
    Image.fromarray(measurement.labels_GM_WM_aligned).save(os.path.join(path_histology_polarimetry_aligned, 
                                                                    'labels_GM_WM_aligned_MLS.png'))


def move_the_alignment_results(measurement: FolderAlignHistology):
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

    # move the folder from to_align to aligned subfolder
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
import os
from PIL import Image
import time
from tqdm import tqdm
import traceback
from collections import defaultdict, Counter
import numpy as np
from rembg import remove
from skimage.measure import regionprops

from histopolalign.helpers import get_folders_lacking_orientation
from histopolalign.helpers import load_wavelengths


def create_the_alignments(histology_path: str, polarimetry_path: str, Verbose: bool = False):
    """
    create_the_alignments is the master function creating the FolderAlignHistology objects for each measurement and histology folder

    Parameters
    ----------
    histology_path : str
        the path to the directory containing the histology folders
    polarimetry_path : str
        the path to the directory containing the polarimetry folders
    verbose : bool
        whether to print the time it takes to load the images (default is False)

    Returns
    -------
    alignment_measurements : list
        the list of the FolderAlignHistology objects
    """
    # get the list of the the polarimetric measurements processed with the new protocol
    folder_paths = get_polarimetry_filenames(polarimetry_path, Verbose = Verbose)    
    start = time.time()
    alignment_measurements = []
    
    # get dictionnaries containing the links between the histology and polarimetry images
    fnames_links = get_link_dicts(histology_path, polarimetry_path)
    print(fnames_links)
    
    # create the FolderAlignHistology objects
    for folder in folder_paths:
        measurement = FolderAlignHistology(folder, fnames_links)
        try:
            
            # check if histology exists for each measurement
            measurement.histology_path
            alignment_measurements.append(measurement)
        except AttributeError:
            pass
    
    end = time.time()
    if Verbose:
        print("Create the alignment objects: {:.3f} seconds.".format(end - start))
        
    return alignment_measurements


def get_polarimetry_filenames(polarimetry_path: str, Verbose: bool = False):
    """
    create_the_alignments is the master function to get the folder names for the polarimetric measurements processed with the new protocol

    Parameters
    ----------
    polarimetry_path : str
        the path to the directory containing the polarimetry folders
    verbose : bool
        whether to print the time it takes to load the images (default is False)

    Returns
    -------
    filtered : list
        the list of the the polarimetric measurements processed with the new protocol
    """
    start = time.time()
    filtered = filter_folder_names(polarimetry_path)
    rename_images(filtered)
    end = time.time()
    if Verbose:
        print("Get polarimetry filenames and rename images: {:.3f} seconds.".format(end - start))
    return filtered


def filter_folder_names(polarimetry_path: list):
    """
    filter_folder_names

    Parameters
    ----------
    polarimetry_path : str
        the path to the directory containing the polarimetric measurements

    Returns
    -------
    filtered : list
        the folder names for the masurements that were processed with the new protocol
    """
    filtered = []
    
    # get the list of the folders into the polarimetry directory
    folder_names = os.listdir(polarimetry_path)
    
    # iterate over the different folders
    for folder in folder_names:
        
        # get the folder number (i.e. the number after HORAO-) and remove the folders that were not processed with the new protocol
        try:
            folder_number = int(folder.split('HORAO-')[-1].split('-')[0])
            # histology is obtained after HORAO-20 and for HORAO-15
            if folder_number >= 20 or folder_number == 15:
                filtered.append(os.path.join(polarimetry_path, folder))
        except ValueError:
            pass
    return filtered


def rename_images(filtered: list):
    """
    function to rename the polarimetric grayscale image with the correct name, normally - should not be used

    Parameters
    ----------
    filtered : list
        the folder names for the masurements that were processed with the new protocol
    """
    # iterate over the different folders
    for folder in filtered:

        # check if it is indeed a folder name
        if folder.split('\\')[-1].startswith('20') and 'HORAO' in folder.split('\\')[-1]:
            
            # iterate over the different wavelengths
            for wavelength in load_wavelengths():
                img_folder = os.path.join(folder, 'polarimetry', wavelength)
                
                # get the list of the files in the polarimetry folder for a wavelength
                for file in os.listdir(img_folder):
                    # if the file is the grayscale image
                    if file.endswith('_realsize.png'):
                        
                        old = os.path.join(folder, 'polarimetry', wavelength, file)
                        fol_short = folder.split('\\')[-1]
                        
                        # create the correct name
                        correct_name = os.path.join(folder, 'polarimetry', wavelength, fol_short + '_' + wavelength + '_realsize.png')
                        try:
                            assert old == correct_name
                        except:
                            if file == '_' + wavelength + '_realsize.png':
                                os.remove(old)
                            else:
                                pass
                                try:
                                    pass
                                    os.rename(old, correct_name)
                                except FileExistsError:
                                    os.remove(old)
                                    pass


def get_link_dicts(histology_path: str, polarimetry_path: str):
    """
    get_link_dicts returns the dictionnaries containing the links between the histology and polarimetry images

    Parameters
    ----------
    histology_path : str
        the path to the directory containing the histology folders
    polarimetry_path : str
        the path to the directory containing the polarimetry folders
    
    Returns
    -------
    fnames_links : dict
        a dictionnary containing the numbers of the slices corresponding to each sample
    """
    fnames_histology, fnames_histology_processed = get_folders_lacking_orientation(histology_path)
    fnames_histo = get_slice_number_dict(fnames_histology + fnames_histology_processed)
    fnames_links = get_histo_polarimetry_link(fnames_histo, polarimetry_path, histology_path)
    return fnames_links


def get_slice_number_dict(fnames_histology: list):
    """
    get_slice_number_dict returns the numbers of the slices corresponding to each sample

    Parameters
    ----------
    fnames_histology : list
        the list of folders containing the histology data

    Returns
    -------
    fnames_histology_link : dict
        the dictionnary to link the polarimetry folders to the corresponding histology folders
    """
    fnames_histo = defaultdict(list)
    
    # iterate over the different folders
    for fname in fnames_histology:
        
        # get the HORAO_idx, the slide_idx and the slice_idx
        HORAO_idx = '-'.join(fname.split('-')[:2])
        slide_idx = int(fname.split('-')[-1].replace('_HE', ''))
        slice_idx = fname.split('-')[-2]
        try:
            int(slice_idx)
        except:
            HORAO_idx = HORAO_idx + '-' + slice_idx

        # add the number of the slice to the list
        fnames_histo[HORAO_idx].append(slide_idx)
    return fnames_histo


def get_histo_polarimetry_link(fnames_histo: list, polarimetry_path: str, histology_path: str):
    """
    function to create a dictionnary to link the histology to the polarimetric measurements

    Parameters
    ----------
    fnames_histo : list
        the number of the slice corresponding to each sample
    polarimetry_path : str
        the path to the folder containing the measurement data
    histology_path : str
        the path to the folder containing the histology data

    Returns
    -------
    fnames_histology_link : dict
        the dictionnary to link the polarimetry folders to the histology folders
    """
    fnames_histology_link = {}
    
    # iterate over the different histology samples
    for horao_idx, slide_idxs in fnames_histo.items():
        
        # get the index of the first and the last slide, as well as the section index for the polarimetry
        idx_before, idx_after = min(slide_idxs), max(slide_idxs)
        idx_section = horao_idx.split('-')[-2]
        idx_slice = horao_idx.split('-')[-1]
                        
        try:
            # iterate over the different files in the polarimetry folder
            for fname in os.listdir(os.path.join(polarimetry_path)):
                if idx_slice in fname and ('-' + idx_section + '-' in fname):
                    # check if the measurement if the high quality one, and if the name has a correct format
                    if not fname.endswith('_1') or '_FR_M' in fname or '-PF_FR' in fname or 'PF-F' in fname:
                        pass
                    else:
                        
                        # check if the measurement was obtained before or after tissue processing
                        before_bool, after_bool = 'BF' in fname, 'AF' in fname
                        
                        # add the folder names to the dictionnary
                        if before_bool:
                            fnames_histology_link[os.path.join(polarimetry_path, fname)] = os.path.join(histology_path, 
                                                                                        horao_idx + '-' + str(idx_before))
                        elif after_bool:
                            fnames_histology_link[os.path.join(polarimetry_path, fname)] = os.path.join(histology_path, 
                                                                                        horao_idx + '-' + str(idx_after))
                        else:
                            raise ValueError
                else:
                    pass
        except:
            traceback.print_exc()

    return fnames_histology_link


class FolderAlignHistology:
    """
    creates an object to align the histology and polarimetry images
    """
    def __init__(self, folder_path: str, histology_paths: dict):
        """
        initialize the object with the folder path and the histology paths

        Parameters
        ----------
        folder_path : str
            the path to the polarimetry folder
        histology_paths : dict
            the dictionnary containing the paths to the histology folders
        """
        # add the histology and the polarimetry paths
        self.folder_path = folder_path
        self.add_histology_path(histology_paths)
        
        # status of the folder - initialized
        self.status = 'initialized'
        
        # initialize the other attributes to None, will be filled later
        self.slide_idx, self.slice_idx, self.HORAO_idx = None, None, None
        self.positions_path, self.annotation_path = None, None
        self.labels_path, self.labels_GM_WM_path = None, None
        self.img_labels, self.img_labels_GM_WM = None, None
        self.polarimetry_path_gs, self.polarimetry_path_gs_650 = None, None
        self.img_polarimetry_gs, self.img_polarimetry_gs_650 = None, None
        self.labels_number, self.labels_number_lab, self.labels_number_lab_GM_WM  = None, None, None

    def __repr__(self):
        """
        print the current status of the folder - print the polarimetry path, the histology path and the status
        """
        f = 'Folder path: {}, histology path : {},  status: {}'.format(self.folder_path, self.histology_path, self.status)
        return f
    
    def add_histology_path(self, histology_paths: dict):
        """
        add_histology_path allows to add the histology path to the object based on the current folder_path

        Parameters
        ----------
        histology_paths : dict
            the dictionnary containing the paths to the histology folders
        """
        # iterate over the different keys and values of the dictionnary and check which one corresponds to the current polarimetry folder
        for key, val in histology_paths.items():
            if key == self.folder_path:
                self.histology_path = val
                
    def _set_images_and_center_of_mass(self, force_recompute: bool = False):
        """
        set_images_and_center_of_mass allows to get the center of mass, histology images, slice number, slice index and HORAO index

        Parameters
        ----------
        force_recompute : bool
            if True, the center of mass and the histology images will be recomputed (default is False)
        """
        self._rename_histology_files()
        self.center_of_mass = self._get_images_and_center_of_mass(force_recompute = force_recompute)
        self.slide_idx, self.slice_idx, self.HORAO_idx = self._get_slice_number()
        self.status = 'histology loaded'
        
    def _rename_histology_files(self):
        """
        rename the filenames to match the standard used

        Parameters
        ----------
        
        Returns
        -------
        """
        old_path = self.histology_path
        
        # if HE was present in the file name, remove it
        new_path = old_path.replace('_HE', '')
        
        # create the new path and change the name of the folder
        new_path = '\\'.join(new_path.split('HORAO')[0].split('\\')[:-1] + ['HORAO' + new_path.split('HORAO')[1]])
        try:
            os.rename(old_path, new_path)
        except PermissionError:
            pass

        return new_path

    def _get_images_and_center_of_mass(self, force_recompute: bool = False):
        """
        load the histology image and get the center of mass (i.e. center of the image) and return it

        Parameters
        ----------
        force_recompute : bool
            if True, the center of mass will be recomputed (default is False)
        
        Returns
        -------
        image : PIL image
            the HE histology image
        center_of_mass : tuple
            the center of mass of the HE histology image
        """
        new_path = self.histology_path
        
        # iterate over the differnt files in the histology folder
        for file in os.listdir(new_path):
            
            # if the file is an image
            if file.endswith('.png'):
                if '-ds.png' in file:
                    fname_new = '-'
                else:
                    fname_new = '-labels'

                # rename the histology files to match the standard if needed
                old_name = os.path.join(new_path, file)
                new_name = os.path.join(new_path, new_path.split('\\')[-1] + fname_new + file.split(fname_new)[-1])
                new_name = new_name.replace('_HE', '')
                try:
                    os.rename(old_name, new_name)
                except PermissionError:
                    pass

        # load the histology image
        path_img = os.path.join(new_path, new_path.split('\\')[-1] + '-ds.png')
        self.histology_img = Image.open(path_img.replace('_HE', ''))

        # path of the pre-computed center of mass
        path_center_of_mass = os.path.join(new_path, 'results', 'center_of_mass.txt')
        
        # if the center of mass has already been computed, load it
        if os.path.exists(path_center_of_mass) and not force_recompute:
            with open(path_center_of_mass) as f:
                lines = f.readlines()
            for idx, line in enumerate(lines):
                lines[idx] = int(line.replace('\n', ''))
            center_of_mass = lines
        else:
            
            # if not, compute it and save it
            x, y = self._get_center_of_mass()
            try:
                os.mkdir(os.path.join(new_path, 'results'))
            except FileExistsError:
                pass
            with open(os.path.join(new_path, 'results', 'center_of_mass.txt'), 'w') as fp:
                for item in [x, y]:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            center_of_mass = [x, y]

        return center_of_mass
    
    def _get_center_of_mass(self):
        """
        gets the center of mass of the histology image

        Parameters
        ----------
        
        Returns
        -------
        x, y : int, int
            the x and y coordinates of the center of mass
        """
        image = self.histology_img
        
        # convert the image to grayscale and binarize it (i.e. 0 if pixel is black, 255 if pixel is white)
        img = np.asarray(remove(image).convert('L'))
        output = np.where(img > 0, 255, 0).astype(int)
        
        # compute the center of mass of the image (i.e. the center of all pixels)
        properties = regionprops(output, img)
        center_of_mass_ = properties[0].centroid
        x, y = int(center_of_mass_[0]), int(center_of_mass_[1])
        return x, y
    
    def _get_slice_number(self):
        """
        gets the slide index, slice index and HORAO index

        Parameters
        ----------
        
        Returns
        -------
        slide_idx, slice_idx, HORAO_idx : int, str, str
            the slide index, slice index and HORAO index
        """
        fname = self.histology_path
        HORAO_idx = fname.split('\\')[-1].split('-')[0]
        slide_idx = int(fname.split('-')[-1].replace('_HE', ''))
        slice_idx = fname.split('-')[-2]
        return slide_idx, slice_idx, HORAO_idx
    
    def _load_images(self):
        """
        load_images is the master function to load the histology images, the polarimetric images, the labels images... and to save
        the paths to the different files of interest

        Parameters
        ----------
        
        Returns
        -------
        """
        # 1. load the alignment corrections
        self.positions_path = os.path.join(self.histology_path, 'annotation/correction_alignment.json')
        
        # 2. load the ROI image
        ROI_found = False
        ROI_path = os.path.join(self.folder_path, 'annotation/ROI.tif')
        while not ROI_found:
            if os.path.exists(ROI_path):
                self.annotation_path = ROI_path
                ROI_found = True
            else:
                print('ROI not found, please create it using ImageJ.')
                input("Press Enter to continue...")

        # 3. load labels image
        self.labels_path = os.path.join(self.histology_path, 'results', 'combined_img.png')
        self.img_labels = Image.open(self.labels_path)
        
        # 4. load labels GM/WM image
        self.labels_GM_WM_path = os.path.join(self.histology_path, 'results', 'combined_img_WM_GM.png')
        self.img_labels_GM_WM = Image.open(self.labels_GM_WM_path)
        
        # 5. load polarimetry grayscale image @550nm
        self.polarimetry_path_gs = os.path.join(self.folder_path, 'polarimetry/550nm/' +
                                                self.folder_path.split('\\')[-1] + '_550nm_realsize.png')
        self.img_polarimetry_gs = Image.open(self.polarimetry_path_gs).convert('L')
    
        # 6. load polarimetry grayscale image @650nm
        self.polarimetry_path_gs_650 = os.path.join(self.folder_path, 'polarimetry/650nm/' + 
                                    self.folder_path.split('\\')[-1] + '_650nm_realsize.png')
        self.img_polarimetry_gs_650 = Image.open(self.polarimetry_path_gs_650).convert('L')
    
        # 7. get the number of different labels
        self.labels_number, self.labels_number_lab, self.labels_number_lab_GM_WM = self._get_labels_numbers()
        
        # 8. apply the function process_image_pathology to the images of interest
        imgs = process_image_pathology(self.histology_img, self.img_labels, self.img_labels_GM_WM, self.center_of_mass)

        # 9. save the images in the FolderAlignHistology object
        self.imgs = imgs
        _, self.histology_cropped, self.labels_cropped, self.labels_GM_WM_cropped = imgs

        self.histology_cropped.save(os.path.join(self.folder_path, 'histology', 'histology_original.png'))
        self.labels_cropped.save(os.path.join(self.folder_path, 'histology', 'labels_original.png'))
        self.labels_GM_WM_cropped.save(os.path.join(self.folder_path, 'histology', 'labels_GM_WM_original.png'))
        
        # 10. change the status of the FolderAlignHistology object
        self.status = 'all images are loaded'
        
    def _get_labels_numbers(self):
        """
        get_labels_numbers returns the number of different labels in the labels image

        Parameters
        ----------
        
        Returns
        -------
        unique_val : int
            the number of different labels in the labels image
        """
        lab = count_pixels(self.img_labels)[0]
        lab_GM_WM = count_pixels(self.img_labels_GM_WM)[0]
        unique_val = max([lab, lab_GM_WM])
        return unique_val, lab, lab_GM_WM

    def _set_the_parameters(self, data):
        """
        _set_the_parameters is the function allowing to set the parameters for the processing of the pathology images

        Parameters
        ----------
        data : dict
            the dictionary containing the parameters of interest
        
        Returns
        -------
        """
        if type(data) == dict:
            self.angle = data['angle']
            self.flip = data['rotation']
            self.shrink = data['shrink']
            self.x_offest = data['x_offest']
            self.y_offest = data['y_offest']
        else:
            self.angle = int(data.rotation.get())
            self.flip = data.flip_txt
            self.shrink = int(data.shrink.get())
            self.x_offest = int(data.x.get())
            self.y_offest = int(data.y.get())
            
            
def load_and_preprocess_imgs(alignment_measurements: list, threshold_min: int = 20, threshold_max: int = 200, force_recompute: bool = False, Verbose: bool = False):
    """
    load_and_preprocess_imgs is the master function to load and pre-process the images of interest

    Parameters
    ----------
    alignment_measurements : list
        the list of the FolderAlignHistology objects
    threshold_min, threshold_max : int, int
        the minimum and maximum threshold values to apply to the histology images (default is 20 and 200)
    force_recompute : bool
        if True, the centers of mass are re-computed (default is False)
    Verbose : bool
        if True, the function prints the time it took to load and pre-process the images (default is False)
    
    Returns
    -------
    alignment_measurements : list
        the updated list of the FolderAlignHistology objects
    """
    
    # iterate over the FolderAlignHistology objects
    for measurement in tqdm(alignment_measurements):
        
        start_all = time.time()
        # get the center of mass, histology images, slice number, slice index and HORAO index
        measurement._set_images_and_center_of_mass(force_recompute = force_recompute)
        end = time.time()
        if Verbose:
            print("Get the images and center of mass: {:.3f} seconds.".format(end - start_all))
        
        start_all = time.time()
        # get the center of mass and load the images
        measurement._load_images()
        end = time.time()
        if Verbose:
            print("Load the images: {:.3f} seconds.".format(end - start_all))
        
    return alignment_measurements
    
    
def process_image_pathology(image: Image, image_labels: Image, image_labels_GM_WM: Image, center_mass: list, path: str = '', angle: int = 0, 
                            flip: str = '', shrink: int = 100, x_offset: int = 0, y_offset: int = 0, selection: bool = False): 
    """
    process_image_pathology is the master function used to transform, resize and crop the polarimetry image as well as the labels image

    Parameters
    ----------
    image : Pillow image
        the histology image 
    image_labels : Pillow image
        the labels image
    image_labels_GM_WM : Pillow image
        the GM/WM labels image to be transformed
    center_mass : list
        the center of the image in the x and y axis
    path : str
        the path to the histology folder for the measurements considered to load an image containing the part of the image to remove ('to_rm_ng.tif', 
        default is '')
    angle : int
        the angle to use to rotate the image (default is 0)
    flip : str
        wether or not the image should be mirrored - 'hv' or 'vh' for vertical and horizonal, 'h' for horizontal, 'v' for vertical 
        and '' for none of the two (default: '')
    shrink : int
        the percentage of the shrink to apply to the image (i.e. > 100 : zoom, < 100 : unzoom, default: 100)
    x_offset, y_offset : int, int
        the x and y offsets to apply to the image (default: 0)
    selection : bool
        if True, the function will not crop the image when calling apply_transformation (default: False)

    Returns
    -------
    images : list of Pillow image
        the transformed H&E and labels images
    """
    # check if the mask for the tissue to analyze exists
    path_mask = os.path.join(path, 'to_rm_ng.tif')
    file_exists = os.path.exists(path_mask)
    
    #  if yes, use it to remove the undesired pixels
    if file_exists:
        mask = np.array(Image.open(path_mask))
        img = np.array(image)
        for idx, row in enumerate(mask):
            for idy, pix in enumerate(row):
                if pix != 0:
                    img[idx, idy] = 0
        image = Image.fromarray(img).convert('RGB')
    
    # apply the transformation (i.e. rotation + flip) to the polarimetric image and the labels images
    [image, image_labels, image_labels_GM_WM] = apply_transformation([image, image_labels, image_labels_GM_WM],center_mass,angle,flip=flip,shrink=shrink,
                                                    x_offset=x_offset, y_offset=y_offset,selection=selection)
        
    # replace the background with white color
    im = Image.new('RGB', image_labels.size, (255, 255, 255))
    im.paste(image_labels, None)
    image_labels = im
    try:
        im = Image.new('RGB', image_labels_GM_WM.size, (255, 255, 255))
        im.paste(image_labels_GM_WM, None)
        image_labels_GM_WM = im
    except:
        pass
    
    img_registration = remove_background_histology(image, image_labels)

    # resize the images to the shape (516,388)
    img_registration_rgb = Image.fromarray(img_registration)
    img_registration_rgb = img_registration_rgb.resize((516, 388), Image.Resampling.NEAREST)
    img_registration = Image.fromarray(img_registration).convert('L')
    img_registration = img_registration.resize((516, 388), Image.Resampling.NEAREST)
    img_labels = image_labels.resize((516, 388), Image.Resampling.NEAREST)
    try:
        image_labels_GM_WM = image_labels_GM_WM.resize((516, 388), Image.Resampling.NEAREST)
    except:
        pass
    
    
    # create the list of images to return
    try:
        images = []
        for img in [img_registration, img_registration_rgb, img_labels, image_labels_GM_WM]:
            mask = np.logical_and(np.array(img.convert('L')) > 20, np.array(img.convert('L')) < 200) * 1
            img = np.array(img)
            if len(img.shape) == 2:
                img[mask == 0] = 0
            else:
                img[mask == 0] = [0, 0, 0]
            images.append(Image.fromarray(img.astype(np.uint8)))
    except:
        images = []
        for img in [img_registration, img_registration_rgb, img_labels]:
            mask = np.logical_and(np.array(img.convert('L')) > 20, np.array(img.convert('L')) < 200) * 1
            img = np.array(img)
            if len(img.shape) == 2:
                img[mask == 0] = 0
            else:
                img[mask == 0] = [0, 0, 0]
            images.append(Image.fromarray(img.astype(np.uint8)))

    return images


def apply_transformation(images: list, center_mass: list, angle_rotation=0, flip='', shrink = 100, x_offset = 0, y_offset = 0, selection: bool = False):
    """
    apply_transformation is the master function to transform the H&E and labels image (rotate, flip, crop and resize)

    Parameters
    ----------
    image : Pillow image
        the image to be transformed
    center_mass : list
        the center of mass of the image
    angle_rotation : int
        the angle to use to rotate the image (default: 0)
    flip : string
        the type of flip to apply to the image (default: '')
    shrink : int
        the percentage of the shrink to apply to the image (i.e. > 100 : zoom, < 100 : unzoom, default: 100)
    x_offset, y_offset : int, int
        the x and y offsets to apply to the image (default: 0)
    selection : bool
        if True, the function will not crop the image (default: False)
        
    Returns
    -------
    imgs_corrected : list
        the list of the transformed images
    """
    if selection:
        imgs_new = images
    else:
        # get the cropped images
        imgs_new = get_cropped_image(images, center_mass)

    imgs_corrected = []
    
    # iterate over the images and apply the transformations
    for img_new in imgs_new:

        # 1. apply the rotation_angle
        if angle_rotation:
            img_new = img_new.rotate(angle_rotation, expand=False)
        else:
            img_new = img_new.rotate(0, expand=False)
        
        # 2. apply the flipping
        if flip == 'vh' or flip == 'hv':
            img_new = img_new.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
            img_new = img_new.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        elif flip == 'v':
            img_new = img_new.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
        elif flip == 'h':
            img_new = img_new.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            pass
        
        # 3. shrink the image
        if shrink != 100:
            old_shape = img_new.size
            new_shape = (int(img_new.size[0] * shrink / 100), int(img_new.size[1] * shrink / 100))
            img_new = img_new.resize(size = new_shape, resample = Image.NEAREST)
            img_new = resize_image_after_shrinkage(img_new, shrink, old_shape)
        else:
            pass

        # 4. move by offset
        img_new = move_by_offset(img_new, x_offset, y_offset)

        imgs_corrected.append(img_new)
    
    return imgs_corrected


def get_cropped_image(images: list, center_mass: list):
    """
    get_cropped_image is used to crop the images around the center of mass

    Parameters
    ----------
    images : list
        the list of images to be cropped
    center_mass : list
        the center of mass of the image
        
    Returns
    -------
    img_arrays : list
        the cropped images
    """
    # set the shape of the cropped image
    x_shape, y_shape = 516 * 2, 388 * 2
    
    # create the list of images as arrays
    img_arrays = []
    for idx, _ in enumerate(images):
        img_arrays.append(np.array(images[idx]))

    # get the min and max values for the cropping
    x_min, x_max = int(center_mass[0] - (x_shape / 2)), int(center_mass[0] + (x_shape / 2) + 1)
    y_min, y_max = int(center_mass[1] - (y_shape / 2)), int(center_mass[1] + (y_shape / 2) + 1)

    # add zeros if the cropping is out of the image
    if y_min < 0:
        to_concat = np.zeros((img_arrays[0].shape[0], np.abs(y_min), 3))
        y_max += np.abs(y_min)
        y_min += np.abs(y_min)
        for idx, img_array in enumerate(img_arrays):
            img_arrays[idx] = np.hstack((to_concat, img_array))
    if y_max > img_arrays[0].shape[1]:
        to_concat = np.zeros((img_arrays[0].shape[0], np.abs(y_max) - img_arrays[0].shape[1], 3))
        for idx, img_array in enumerate(img_arrays):
            img_arrays[idx] = np.hstack((img_array, to_concat))

    if x_min < 0:
        to_concat = np.zeros((np.abs(x_min), img_arrays[0].shape[1], 3))
        x_max += np.abs(x_min)
        x_min += np.abs(x_min)
        for idx, img_array in enumerate(img_arrays):
            img_arrays[idx] = np.vstack((to_concat, img_array))
    if x_max > img_arrays[0].shape[0]:
        to_concat = np.zeros((np.abs(x_max) - img_arrays[0].shape[0], img_arrays[0].shape[1], 3))
        for idx, img_array in enumerate(img_arrays):
            img_arrays[idx] = np.vstack((img_array, to_concat))
    
    # actually crop the images
    for idx, img_array in enumerate(img_arrays):
        
        img = img_array[x_min:x_max-1, y_min:y_max-1].astype(np.uint8)
        img_arrays[idx] = Image.fromarray(img).rotate(-90, expand=True).resize((516, 388), Image.Resampling.NEAREST)
        
    return img_arrays


def resize_image_after_shrinkage(img: Image, shrink: int, old_shape: tuple):
    """
    resize_image_after_shrinkage is used to resize the image to the original one by cropping it after zooming / unzooming

    Parameters
    ----------
    img : PIL image
        the image to be resized
    shrink : int
        the percentage of the shrink applied to the image
    old_shape : tuple
        the shape of the original image
    
    Returns
    -------
    img : PIL image
        the resized image
    """
    if shrink > 100:
        # get the min and max values for the cropping
        x_min = round((img.size[1] - old_shape[1]) / 2)
        x_max = x_min + old_shape[1]
        y_min = round((img.size[0] - old_shape[0]) / 2)
        y_max = y_min + old_shape[0]
        img = np.asarray(img)
        
        # do the cropping and return the image
        return Image.fromarray(img[x_min:x_max, y_min:y_max].astype(np.uint8))
    else:
        img = np.asarray(img)
        assert shrink < 100
        
        # add zeros as the size of the new image is smaller than the original one
        to_concat = np.zeros((img.shape[0], round((old_shape[0]-img.shape[1])/2), 3))  
        img = np.hstack((to_concat, img, to_concat))
        to_concat = np.zeros((round((old_shape[1]-img.shape[0])/2), img.shape[1], 3))  
        img = np.vstack((to_concat, img, to_concat))
        
        # return the image
        return Image.fromarray(img.astype(np.uint8))
    

def move_by_offset(img, x_offset, y_offset):
    """
    move_by_offset is used to move the image by the given offset

    Parameters
    ----------
    img : PIL image
        the image to be resized
    x_offset, y_offset : int, int
        the x and y offsets to apply to the image (default: 0)
    
    Returns
    -------
    img : PIL image
        the image moved by the given offset
    """
    x_offset, y_offset = - y_offset, - x_offset
    img = np.asarray(img)
        
    # add zeros to the image to allow further cropping for the x-axis...
    if x_offset > 0:
        to_concat = np.zeros((int(x_offset), img.shape[1], 3))
        img = np.vstack((img, to_concat))
    elif x_offset < 0:
        to_concat = np.zeros((int(np.abs(x_offset)), img.shape[1], 3))
        img = np.vstack((to_concat, img))
    
    # ... and the y-axis
    if y_offset > 0:
        to_concat = np.zeros((img.shape[0], int(y_offset), 3))
        img = np.hstack((img, to_concat))
    elif y_offset < 0:
        to_concat = np.zeros((img.shape[0], int(np.abs(y_offset)), 3))
        img = np.hstack((to_concat, img))
    
    # get the min and max values for the cropping
    x_min, x_max = int(max(x_offset, 0)), int(min(img.shape[0], img.shape[0] + x_offset))
    y_min, y_max = int(max(y_offset, 0)), int(min(img.shape[1], img.shape[1] + y_offset))
    
    # do the cropping and return the image
    img_resized = img[x_min:x_max, y_min:y_max]
    return Image.fromarray(img_resized.astype(np.uint8))


def remove_background_histology(image: Image, image_labels: Image):
    """
    remove_background_histology is used to remove the background of the histology H&E image

    Parameters
    ----------
    image : Pillow image
        the histology H&E image 
    image_labels : Pillow image
        the labels image

    Returns
    -------
    img_registration : np.array
        the H&E image with the background removed
    """
    # create the mask to mask the background for the histology image...
    mask = np.where(np.array(image_labels).sum(axis = 2) == 0)
    img_registration = np.array(image)

    for idx, idy in zip(mask[0], mask[1]):
        img_registration[idx, idy] = [0, 0, 0]
        
    return img_registration


def count_pixels(matrix: Image):
    """
    count_pixels returns the number of different labels one image

    Parameters
    ----------
    matrix : Image
        the image of interest
        
    Returns
    -------
    len(keys_signal) : int
        the number of different labels in the image
    """
    all_pixels = []
        
    # convert the image to an array and downsample the image to faster processing
    matrix_arr = np.array(matrix)
    matrix_arr = matrix_arr[::5, ::5, :]
        
    # iterate over the image    
    for x in matrix_arr:
        for y in x:
            # and append the pixel to the list of pixels
            all_pixels.append(tuple(y))
                
    # get the number of different labels
    keys = list(Counter(all_pixels).keys())
    keys_signal = [key for key in keys if key != (255, 255, 255) and key != (0, 0, 0)]
    return len(keys_signal), keys_signal
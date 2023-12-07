import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import circstd
from tqdm import tqdm
import os
import matplotlib

def get_parameters_healthy(polarimetry_path: str, azimuth_sq_size: int = 8, type_ = ''):
    """
    get_parameters_healthy is a function used to collect the polarimetric parameters of the healthy tissue in the white and in the grey matter

    Parameters
    ----------
    polarimetry_path : str
        the path to the folder containing the polarimetric images
    azimuth_sq_size : int, optional, default is 8
        the size of the square used to compute the local azimuth standard deviation
        
    Returns
    ----------
    values_folder : dict of dict
        the dictionary containing the polarimetric parameters of the healthy tissue in the white and in the grey matter
    """     
    # initialize the dictionary that will contain the polarimetric parameters
    parameters = ['totP', 'linR']
    values_folder = {'totP': defaultdict(list), 'linR': defaultdict(list), 'azimuth': defaultdict(list)}
    
    # loop over the folders containing the polarimetric images
    for folder in tqdm(os.listdir(polarimetry_path)):
        
        if type_ in folder:
            # load the Mueller matrix
            path_MM = os.path.join(polarimetry_path, folder, 'polarimetry', '550nm', 'MM.npz')
            MM = np.load(path_MM)
            totP = MM['totP']
            linR = MM['linR']
            azimuth = MM['azimuth']
            mask = MM['Msk']
            
            # load the grey / white matter annotation mask
            path_annotation = os.path.join(polarimetry_path, folder, 'annotation', 'merged.png')
            img = np.array(Image.open(path_annotation))
            
            # loop over the pixels and add them to the values dictionary if they are not background
            for idx, x in enumerate(img):
                for idy, y in enumerate(x):
                    if y == 0 or not mask[idx, idy]:
                        pass
                    else:
                        for mat, param in zip([totP, linR], parameters):
                            values_folder[param][y].append(mat[idx, idy])
            
            # do the same for the azimuth of the optical axis
            for idx, x in enumerate(img):
                for idy, _ in enumerate(x):
                    
                    # get the neighbors of the pixel
                    neighbors = img[idx : idx + azimuth_sq_size, idy : idy + azimuth_sq_size]
                    msked = mask[idx : idx + azimuth_sq_size, idy : idy + azimuth_sq_size]
                    
                    # check is most pixels are valid
                    if sum(msked.flatten()) > 0.5 * azimuth_sq_size * azimuth_sq_size:
                        
                        # check if all the pixels belongs to the same class
                        if np.sum(neighbors == 128) == azimuth_sq_size * azimuth_sq_size or np.sum(neighbors == 255) == azimuth_sq_size * azimuth_sq_size or np.sum(neighbors == 0) == azimuth_sq_size * azimuth_sq_size:
                            # get the azimuth of the neighbors
                            neighbors_MM = azimuth[idx : idx + azimuth_sq_size, idy : idy + azimuth_sq_size]
                            neighbors_cleaned = []
                            
                            # add the valid neighbors to the list
                            for idx_msk, x_msk in enumerate(msked):
                                for idy_msk, y_msk in enumerate(x_msk):
                                    if y_msk:
                                        neighbors_cleaned.append(neighbors_MM[idx_msk, idy_msk])
                                    else:
                                        pass
                            
                            # add the value of the circular standard deviation to the dictionary
                            if np.sum(neighbors == 128) == azimuth_sq_size * azimuth_sq_size:
                                values_folder['azimuth'][128].append(circstd(neighbors_cleaned, high=180, low=0))
                            elif np.sum(neighbors == 255) == azimuth_sq_size * azimuth_sq_size:
                                values_folder['azimuth'][255].append(circstd(neighbors_cleaned, high=180, low=0))
                            elif np.sum(neighbors == 0) == azimuth_sq_size * azimuth_sq_size:
                                values_folder['azimuth'][0].append(circstd(neighbors_cleaned, high=180, low=0))
                        
    return values_folder



def get_all_folders(neoplastic_polarimetry_path: str, type_ = ''):
    """
    get_all_folders finds all the neoplastic tumor polarimetric measurements folders

    Parameters
    ----------
    neoplastic_polarimetry_path : str
        the path to the folder containing the polarimetric images of the neoplastic tissue
        
    Returns
    ----------
    path_folders : list
        the list containing the paths to the neoplastic tumor polarimetric measurements folders
    """     
    path_folders = []
    
    # iterate over the patient folders
    for folder in os.listdir(neoplastic_polarimetry_path):
        folder_measurement = os.path.join(neoplastic_polarimetry_path, folder)
            
        # iterate over the measurement folders
        for measurement in os.listdir(folder_measurement):
            if type_ in measurement:
                path_folders.append(os.path.join(folder_measurement, measurement))
    return path_folders



def get_the_values(path_folders: list, azimuth_sq_size: int = 8):
    """
    get_the_values loads the polarimetric parameters of the neoplastic tissue in the white and in the grey matter

    Parameters
    ----------
    path_folders : list
        the list containing the paths to the neoplastic tumor polarimetric measurements folders
    azimuth_sq_size : int, optional, default is 8
        the size of the square used to compute the local azimuth standard deviation
        
    Returns
    ----------
    values_folder : dict of dict
        the dictionary containing the polarimetric parameters of the neoplastic tissue in the white and in the grey matter
    """     
    # initiate the dictionary used to return the polarimetric parameters
    values_folder = {'totP': {(153,153,0): {(255, 0, 0): [], (0, 255, 255): [], (0, 0, 255): []}, 
                              (153,77, 0): {(255, 0, 0): [], (0, 255, 255): [], (0, 0, 255): []}},
                    'linR': {(153,153,0): {(255, 0, 0): [], (0, 255, 255): [], (0, 0, 255): []}, 
                              (153,77, 0): {(255, 0, 0): [], (0, 255, 255): [], (0, 0, 255): []}},
                    'azimuth': {(153,153,0): {(255, 0, 0): [], (0, 255, 255): [], (0, 0, 255): []}, 
                              (153,77, 0): {(255, 0, 0): [], (0, 255, 255): [], (0, 0, 255): []}}}
    
    # iterate over the folders containing the polarimetric images
    for path_folder in tqdm(path_folders):

        # load the annotations
        path_labels = os.path.join(path_folder, 'histology', 'labels_augmented_masked.png')
        labels = np.array(Image.open(path_labels))
        path_labels_GM_WM = os.path.join(path_folder, 'histology', 'labels_augmented_GM_WM_masked.png')
        labels_GM_WM = np.array(Image.open(path_labels_GM_WM))

        # load the Mueller matrix
        path_MM = os.path.join(path_folder, 'polarimetry', '550nm', 'MM.npz')
        MM = np.load(path_MM)
        parameters = ['totP', 'linR']
        totP = MM['totP']
        linR = MM['linR']
        azimuth = MM['azimuth']
        mask = MM['Msk']

        # iterate over the pixels
        for idx, x in enumerate(labels):
            for idy, y in enumerate(x):
                # check if the pixel is background or if it should be masked
                if sum(y) == 0 or sum(y) == 3 * 255 or not mask[idx, idy]:
                    pass
                else:
                    
                    # get the depolarization and retardance values
                    labels_GM_WM_pixel = labels_GM_WM[idx, idy]
                    labels_TCC_pixel = y
                    try:
                        for mat, param in zip([totP, linR], parameters):
                            values_folder[param][tuple(labels_GM_WM_pixel)][tuple(labels_TCC_pixel)].append(mat[idx, idy])
                    except:
                        # try to correct the labels, if needed
                        if abs(np.sum(labels_GM_WM_pixel - [153, 77, 0])) < 20:
                            labels_GM_WM_pixel = [153, 77, 0]
                        elif abs(np.sum(labels_GM_WM_pixel - [153, 153, 0])) < 20:
                            labels_GM_WM_pixel = [153, 153, 0]

                        if abs(np.sum(labels_TCC_pixel - [255, 0, 0])) < 20:
                            labels_TCC_pixel = [255, 0, 0]
                        elif abs(np.sum(labels_TCC_pixel - [0, 255, 255])) < 20:
                            labels_TCC_pixel = [0, 255, 255]
                        elif abs(np.sum(labels_TCC_pixel - [0, 0, 255])) < 20:
                            labels_TCC_pixel = [0, 0, 255]
                        else:
                            pass
                        
                        # try again with the corrected labels
                        try:
                            for mat, param in zip([totP, linR], parameters):
                                values_folder[param][tuple(labels_GM_WM_pixel)][tuple(labels_TCC_pixel)].append(mat[idx, idy])
                        except:
                            pass

                                
                # azimuth
                neighbors = labels_GM_WM[idx : idx + azimuth_sq_size, idy : idy + azimuth_sq_size]
                neighbors_TCC = labels[idx : idx + azimuth_sq_size, idy : idy + azimuth_sq_size]
                msked = mask[idx : idx + azimuth_sq_size, idy : idy + azimuth_sq_size]
                
                # check is most pixels are valid
                if sum(msked.flatten()) > 0.5 * azimuth_sq_size * azimuth_sq_size:

                    # check if all the pixels belongs to the same class
                    if np.sum(neighbors == [153, 77, 0]) == azimuth_sq_size * azimuth_sq_size * neighbors.shape[2] or np.sum(neighbors == [153, 153, 0]) == azimuth_sq_size * azimuth_sq_size * neighbors.shape[2]:
                        neighbors_cleaned = azimuth[idx : idx + azimuth_sq_size, idy : idy + azimuth_sq_size]
                            
                        GM_WM = None
                        TCC = None
                            
                        if np.sum(neighbors == [153, 77, 0]) == azimuth_sq_size * azimuth_sq_size * neighbors.shape[2]:
                            GM_WM = [153, 77, 0]
                        elif np.sum(neighbors == [153, 153, 0]) == azimuth_sq_size * azimuth_sq_size * neighbors.shape[2]:
                            GM_WM = [153, 153, 0]
                                
                        if np.sum(neighbors_TCC == [255, 0, 0]) == azimuth_sq_size * azimuth_sq_size * neighbors_TCC.shape[2]:
                            TCC = [255, 0, 0]
                        elif np.sum(neighbors_TCC == [0, 255, 255]) == azimuth_sq_size * azimuth_sq_size * neighbors_TCC.shape[2]:
                            TCC = [0, 255, 255]
                        elif np.sum(neighbors_TCC == [0, 0, 255]) == azimuth_sq_size * azimuth_sq_size * neighbors_TCC.shape[2]:
                            TCC = [0, 0, 255]
                            
                        if GM_WM is None or TCC is None:
                            pass
                        else:
                            
                            # add the value of the circular standard deviation to the dictionary
                            values_folder['azimuth'][tuple(GM_WM)][tuple(TCC)].append(circstd(neighbors_cleaned, high=180, low=0))
                        
    return values_folder



def plot_one_histogram(values: dict, ax: matplotlib.axes, idx_param: int, n_bins_azimuth: int, n_bins: int, ranged: list, idx_plt: int, 
                       x_txt: float, y_txt: float, color: str, label: str, txt: str):
    """
    plot_one_histogram is used to plot the distribution of one polarimetric parameter in one tissue class

    Parameters
    ----------
    values : dict
        the dictionnary containing the polarimetric parameters 
    ax : axis
        the axis on which the histogram will be plotted
    idx_param : int
        the index of the polarimetric parameter to plot
    n_bins_azimuth : int
        the number of bins to use to plot the azimuth
    n_bins : int 
        the number of bins to use to plot the other polarimetric parameters
    ranged : list
        the limits of the histogram plot on the x axis
    idx_plt : int
        the index of the polarimetric parameter to plot
    x_txt, y_txt : float, float
        the x and y index of where the text should be displayed
    color : str
        the color to plot the histogram and display the text
    label : str
        the label to use for the legend
    txt : str
        the text to display
    """   
    # get the histogram values
    if idx_plt == 2:
        y, x = np.histogram(values[idx_param], bins=n_bins_azimuth, density=True, range=ranged)
    else:
        y, x = np.histogram(values[idx_param], bins=n_bins, density=True, range=ranged)
        
    # assert a correct number of bins
    x_plot = []
    for idx, _ in enumerate(x):
        try: 
            x_plot.append((x[idx] + x[idx + 1]) / 2)
        except:
            if idx_plt != 2:
                assert len(x_plot) == n_bins
            else:
                assert len(x_plot) == n_bins_azimuth
                    
    # normalize the histogram
    y = y / np.max(y)
    # and plot it
    wm, = ax.plot(x_plot, y, c = color, label = label, linewidth=3)
    
    # get the mean, std and median of the distribution
    mean = np.nanmean(values[idx_param])
    std = np.nanstd(values[idx_param])
    median = np.nanmedian(values[idx_param])
        
    # add the text to the plot
    ax.text(x_txt, y_txt, txt.format(mean = mean, std = std, median = median), fontsize=28, fontweight = 'bold', c = color)
    
        
        
def plot_histograms_paper(values_healthy: dict, tumor_values: dict, params: list, GM_WM: tuple, path_save: str, WM: bool = False):
    """
    plot_histograms_paper is used to plot the combination of the histograms of the polarimetric parameters for the manuscript

    Parameters
    ----------
    values_healthy : dict
        the dictionary containing the polarimetric parameters of the healthy tissue
    tumor_values : dict
        the dictionary containing the polarimetric parameters of the neoplastic tissue
    params : list
        the list of the polarimetric parameters to plot
    GM_WM : tuple
        the tuple containing the RGB code of the tissue class to plot
    path_save : str
        the path to the folder where the plots will be saved
    WM : bool, optional, default is False
        indicates wether the tissue class is the white or the grey matter
    """
    fig, axs = plt.subplots(3, 1, figsize = (15, 19))
    
    # set the number of bins to use
    n_bins = 60
    n_bins_azimuth = 30
    
    # iterate over the polarimetric parameters
    for idx_plt, (param, ax, ranged) in enumerate(zip(params, axs, [[0.5, 1], [0,40], [0,70]])):
        
        # create the plot
        ax.axis(ymin=0,ymax=1.5)
        ax.locator_params(axis='y', nbins=4)
        ax.locator_params(axis='x', nbins=5)
        
        # generate the text formatter to display for healthy tissue
        if idx_plt == 0:
            x_txt, y_txt = 0.5, 1.1
            txt = "μ: {mean:.2f}\nσ: {std:.2f}\nm: {median:.2f}"
        elif idx_plt == 1:
            x_txt, y_txt = 0.5, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        elif idx_plt == 2:
            x_txt, y_txt = 0.5, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        
        # plot the histogram of the healthy tissue
        if WM:
            plot_one_histogram(values_healthy[param], ax, 255, n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt, 
                            'green', 'TF WM', txt)
        else:
            plot_one_histogram(values_healthy[param], ax, 128, n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt, 
                            'green', 'TF GM', txt)
        
        # generate the text formatter to display for neoplastic tissue with proportion 70-100%, and plot the distribution
        if idx_plt == 0:
            x_txt, y_txt = 0.64, 1.1
            txt = "μ: {mean:.2f}\nσ: {std:.2f}\nm: {median:.2f}"
        elif idx_plt == 1:
            x_txt, y_txt = 11, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        elif idx_plt == 2:
            x_txt, y_txt = 19.5, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        plot_one_histogram(tumor_values[param][GM_WM], ax, (255,0,0), n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt,
                          'red', '70-100%', txt)
        
        # generate the text formatter to display for neoplastic tissue with proportion 30-70%, and plot the distribution
        if idx_plt == 0:
            x_txt, y_txt = 0.78, 1.1
            txt = "μ: {mean:.2f}\nσ: {std:.2f}\nm: {median:.2f}"
        elif idx_plt == 1:
            x_txt, y_txt = 22, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        elif idx_plt == 2:
            x_txt, y_txt = 38, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
            
        plot_one_histogram(tumor_values[param][GM_WM], ax, (0,0,255), n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt,
                          'blue', '30-70%', txt)
        
        # generate the text formatter to display for neoplastic tissue with proportion 0-30%, and plot the distribution
        if idx_plt == 0:
            x_txt, y_txt = 0.92, 1.1
            txt = "μ: {mean:.2f}\nσ: {std:.2f}\nm: {median:.2f}"
        elif idx_plt == 1:
            x_txt, y_txt = 33, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        elif idx_plt == 2:
            x_txt, y_txt = 57, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
            
        plot_one_histogram(tumor_values[param][GM_WM], ax, (0,255,255), n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt,
                          'deepskyblue', '0-30%', txt)

        
        # plot the value of the noise in the azimuth in the background
        if idx_plt == 2:
            x_txt, y_txt = 57, 0.7
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
            plot_one_histogram(values_healthy[param], ax, 0, n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt, 
                           'black', 'BG', txt)
        
        # create the legend
        legend_properties = {'weight':'bold', 'size': 30}
        
        if idx_plt == 0:
            leg = ax.legend(loc='best', bbox_to_anchor=(0.985, 0.6, 0, 0), prop=legend_properties)
            leg.get_frame().set_edgecolor('white')
        else:
            leg = ax.legend(loc='best', bbox_to_anchor=(1.40, 0.6, 0, 0), prop=legend_properties)
            leg.get_frame().set_edgecolor('white')
        
        # change the font size of the ticks
        ax.tick_params(axis='both', which='major', labelsize=18)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(26)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(26)
            tick.label1.set_fontweight('bold')
            
    # save the plots
    plt.tight_layout()
    fig.savefig(path_save)
    fig.savefig(path_save.replace('.png', '.pdf'))
    
    plt.close()
    
    
    
def combine_the_values(values_folder: dict):
    """
    combine_the_values is used to combine the values of the polarimetric parameters of the neoplastic tissue

    Parameters
    ----------
    values_folder : dict
        the dictionary containing the polarimetric parameters of the neoplastic tissue
    """
    # initialize the dictionary that will contain the combined values
    values_folder_combined = {'totP': defaultdict(list),
                          'linR': defaultdict(list), 
                          'azimuth': defaultdict(list)}
    
    # iterate over the polarimetric parameters and combine the values
    for param, values in values_folder.items():
        for gm_wm, val in values.items():
            for _, va in val.items():
                for v in va:
                    try:
                        print(len(v))
                    except:   
                        values_folder_combined[param][gm_wm].append(v)
    
    return values_folder_combined



def plot_histograms_paper_combined(values_healthy: dict, values_folder_combined: dict, params: list, WM: bool, path_save: str):
    """
    plot_histograms_paper_combined is used to plot the combination of the histograms of the polarimetric parameters for the manuscript when combining the neoplastic tissue measurements altogether

    Parameters
    ----------
    values_healthy : dict
        the dictionary containing the polarimetric parameters of the healthy tissue
    values_folder_combined : dict
        the dictionary containing the polarimetric parameters of the neoplastic tissue combined together
    params : list
        the list of the polarimetric parameters to plot
    WM : bool
        indicates wether the tissue class is the white or the grey matter
    path_save : str
        the path to the folder where the plots will be saved
    """
    fig, axs = plt.subplots(3, 1, figsize = (15, 19))
    
    n_bins = 60
    n_bins_azimuth = 30
    
    # iterate over the polarimetric parameters
    for idx_plt, (param, ax, ranged) in enumerate(zip(params, axs, [[0.5, 1], [0,40], [0,70]])):
        
        # create the plot and configure it
        ax.axis(ymin=0,ymax=1.5)
        ax.locator_params(axis='y', nbins=4)
        ax.locator_params(axis='x', nbins=5)
        if idx_plt == 0:
            x_txt, y_txt = 0.5, 1.1
            txt = "μ: {mean:.2f}\nσ: {std:.2f}\nm: {median:.2f}"
        elif idx_plt == 1:
            x_txt, y_txt = 0.5, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        elif idx_plt == 2:
            x_txt, y_txt = 0.5, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        
        # plot the histogram of the healthy tissue
        if not WM:
            plot_one_histogram(values_healthy[param], ax, 128, n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt, 
                            'green', 'TF GM', txt)
        else:
            plot_one_histogram(values_healthy[param], ax, 255, n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt, 
                            'green', 'TF WM', txt)
        
        # plot the histograms of the neoplastic tissue
        if idx_plt == 0:
            x_txt, y_txt = 0.64, 1.1
            txt = "μ: {mean:.2f}\nσ: {std:.2f}\nm: {median:.2f}"
        elif idx_plt == 1:
            x_txt, y_txt = 11, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        elif idx_plt == 2:
            x_txt, y_txt = 19.5, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"

        if not WM:
            plot_one_histogram(values_folder_combined[param], ax, (153,77,0), n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt,
                            'red', 'NP GM', txt)
        else:
            plot_one_histogram(values_folder_combined[param], ax, (153,153,0), n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt,
                            'red', 'NP WM', txt)


        # plot the value of the noise in the azimuth in the background
        if idx_plt == 2:
            x_txt, y_txt = 57, 0.7
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
            
            plot_one_histogram(values_healthy[param], ax, 0, n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt, 
                           'black', 'BG', txt)
            
        
        # create the legend
        legend_properties = {'weight':'bold', 'size': 30}
        
        if idx_plt == 0:
            leg = ax.legend(loc='best', bbox_to_anchor=(1.02, 0.6, 0, 0), prop=legend_properties)
            leg.get_frame().set_edgecolor('white')
        else:
            leg = ax.legend(loc='best', bbox_to_anchor=(1.37, 0.6, 0, 0), prop=legend_properties)
            leg.get_frame().set_edgecolor('white')
        
        ax.tick_params(axis='both', which='major', labelsize=18)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(26)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(26)
            tick.label1.set_fontweight('bold')
            
    # and save the plot
    plt.tight_layout()
    fig.savefig(path_save)
    fig.savefig(path_save.replace('.png', '.pdf'))
    plt.close()
    
    
    
def get_number_of_samples_tcp(path_folders: list):
    """
    get_number_of_samples_tcp is used to get the number of samples showing every tumor cell proportion

    Parameters
    ----------
    path_folders : list
        the list containing the paths to the neoplastic tumor polarimetric measurements folders
    """
    numbers = {'0-30': 0, '30-70': 0, '70-100': 0}

    for folder in path_folders:
        # load the labels
        mask = np.asarray(Image.open(os.path.join(folder, 'histology', 'labels_augmented_masked.png')))
        
        # and check if the images contains over 800 pixels of each tumor cell class
        if np.sum(np.sum(mask == [0, 0, 255], axis = 2) == 3) > 800:
            numbers['30-70'] += 1
        if np.sum(np.sum(mask == [0, 255, 255], axis = 2) == 3) > 800:
            numbers['0-30'] += 1
        if np.sum(np.sum(mask == [255, 0, 0], axis = 2) == 3) > 800:
            numbers['70-100'] += 1
            
    return numbers



def plot_histograms_methods(values_healthy: dict, tumor_values: dict, params: list, WM: bool, path_save: str):
    """
    plot_histograms_methods is used to plot the histograms used to generate the figure in the methods

    Parameters
    ----------
    values_healthy : dict
        the dictionary containing the polarimetric parameters for one healthy tissue measurement
    tumor_values : dict
        the dictionary containing the polarimetric parameters for one neoplastic tissue measurement
    params : list
        the list of the polarimetric parameters to plot
    GM_WM : tuple
        the tuple containing the RGB code of the tissue class to plot
    WM : bool
        indicates wether the tissue class is the white or the grey matter
    path_save : str
        the path to the folder where the plots will be saved
    """
    fig, axs = plt.subplots(3, 1, figsize = (15, 19))
    
    n_bins = 60
    n_bins_azimuth = 30
    
    for idx_plt, (param, ax, ranged) in enumerate(zip(params, axs, [[0.5, 1], [0,40], [0,70]])):
        
        # configure the plot
        ax.axis(ymin=0,ymax=1.5)
        ax.locator_params(axis='y', nbins=4)
        ax.locator_params(axis='x', nbins=5)
        
        # plot the histogram of the healthy tissue
        if idx_plt == 0:
            x_txt, y_txt = 0.5, 1.1
            txt = "μ: {mean:.2f}\nσ: {std:.2f}\nm: {median:.2f}"
        elif idx_plt == 1:
            x_txt, y_txt = 0.5, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        elif idx_plt == 2:
            x_txt, y_txt = 0.5, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        plot_one_histogram(values_healthy[param], ax, 255, n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt, 
                            'darkgreen', 'TF WM', txt)
        
        if idx_plt == 0:
            x_txt, y_txt = 0.64, 1.1
            txt = "μ: {mean:.2f}\nσ: {std:.2f}\nm: {median:.2f}"
        elif idx_plt == 1:
            x_txt, y_txt = 11, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        elif idx_plt == 2:
            x_txt, y_txt = 19.5, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        plot_one_histogram(values_healthy[param], ax, 128, n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt, 
                            'lime', 'TF GM', txt)
        

        # plot the histogram of the neoplastic tissue
        if idx_plt == 0:
            x_txt, y_txt = 0.78, 1.1
            txt = "μ: {mean:.2f}\nσ: {std:.2f}\nm: {median:.2f}"
        elif idx_plt == 1:
            x_txt, y_txt = 20.5, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        elif idx_plt == 2:
            x_txt, y_txt = 38.5, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        
        plot_one_histogram(tumor_values[param][(153, 153, 0)], ax, (255,0,0), n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt,
                          'red', '70-100% WM', txt)

        if idx_plt == 0:
            x_txt, y_txt = 0.92, 1.1
            txt = "μ: {mean:.2f}\nσ: {std:.2f}\nm: {median:.2f}"
        elif idx_plt == 1:
            x_txt, y_txt = 32, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
        elif idx_plt == 2:
            x_txt, y_txt = 57, 1.1
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"

        plot_one_histogram(tumor_values[param][(153, 77, 0)], ax, (0,255,255), n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt,
                          'turquoise', '0-30% GM', txt)
            
            
        # plot the value of the noise in the azimuth in the background
        if idx_plt == 2:
            x_txt, y_txt = 57, 0.7
            txt = "μ: {mean:.1f}\nσ: {std:.1f}\nm: {median:.1f}"
            
            plot_one_histogram(values_healthy[param], ax, 0, n_bins_azimuth, n_bins, ranged, idx_plt, x_txt, y_txt, 
                           'black', 'BG', txt)
        
        # create the legend
        legend_properties = {'weight':'bold', 'size': 26}
        
        if idx_plt == 0:
            leg = ax.legend(loc='best', bbox_to_anchor=(0.985, 0.6, 0, 0), prop=legend_properties)
            leg.get_frame().set_edgecolor('white')
        elif idx_plt == 1:
            leg = ax.legend(loc='best', bbox_to_anchor=(1, 0.6, 0, 0), prop=legend_properties)
            leg.get_frame().set_edgecolor('white')

        
        ax.tick_params(axis='both', which='major', labelsize=18)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(26)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(26)
            tick.label1.set_fontweight('bold')
        else:
            leg = ax.legend(loc='best', bbox_to_anchor=(1, 0.6, 0, 0), prop=legend_properties)
            leg.get_frame().set_edgecolor('white')
            
    # and save the plot
    plt.tight_layout()
    fig.savefig(path_save)
    fig.savefig(path_save.replace('.pdf', '.png'))
    plt.close()
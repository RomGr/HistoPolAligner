import os
from tqdm import tqdm

import numpy as np
from scipy.stats import circstd

from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.cbook
import matplotlib.colors as clr

import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)



def get_and_plots_stds(measurements: list, sq_size: int = 4):
    """
    get_and_plots_stds is the master function to create the plots to visualize the azimuth noise

    Parameters
    ----------
    measurements : list
        the list of the path to the measurements folders

    Returns
    -------
    sq_size : int
        the size of the square to compute the azimuth std (default is 4)
    """
    azimuth_stds = {}

    # iterate over the measurements
    for folder in tqdm(measurements):
        
        # load the azimuth
        path_polarimetry = os.path.join(folder, 'polarimetry', '550nm')
        azimuth = np.load(os.path.join(path_polarimetry, 'MM.npz'))['azimuth']
        
        # initialize the azimuth std
        azimuth_std = np.zeros((round(azimuth.shape[0]), round(azimuth.shape[1])))
        
        # iterate over the pixels
        for idx in range(0, len(azimuth), 1):
            for idy in range(0, len(azimuth[0]), 1):
                try:
                    # extract the azimuth in a square around the pixel and compute the std
                    assert azimuth[idx-sq_size//2:idx+sq_size//2,idy-sq_size//2:idy+sq_size//2].shape == ((4,4))
                    std = circstd(azimuth[idx-sq_size//2:idx+sq_size//2,idy-sq_size//2:idy+sq_size//2], high=180)
                    try:
                        azimuth_std[idx, idy] = std
                    except:
                        pass
                except:
                    pass
        
        # remove the std > 40 for visualization purposes
        azimuth_std[azimuth_std > 40] = 40
        azimuth_stds[folder] = azimuth_std
        
        # load the GM/WM mask and plot the azimuth noise
        try:
            path_mask = os.path.join(folder, 'histology', 'labels_augmented_GM_WM_masked.png')
            mask = np.asarray(Image.open(path_mask))
            plot_azimuth_noise(azimuth_std, folder, mask)
        except:
            path_mask = os.path.join(folder, 'annotation', 'merged.png')
            mask = np.asarray(Image.open(path_mask))
            plot_azimuth_noise(azimuth_std, folder, mask, healthy= True)
                
        plt.close()
        
    return azimuth_stds


def plot_azimuth_noise(azimuth_std: np.array, folder: str, mask: np.array, healthy: bool = False):
    """
    plot_azimuth_noise is the function to plot the azimuth noise

    Parameters
    ----------
    azimuth_std : np.array
        the azimuth circular standard deviation matrix
    folder : str
        the path to the measurement folder
    mask : np.array
        the mask of the GM / WM
    healthy : bool
        whether the measurement is healthy tissue or not

    Returns
    -------
    """
    # get the colormaps for the legend
    n_bins = 200
    colors = [[0, 0, 0.5], [0, 0, 1], 
              [0, 0.5, 1], [0, 1, 1], 
              [0.5, 1, 0.5], [1, 1, 0], 
              [1, 0.5, 0], [1, 0, 0], 
              [0.5, 0, 0]]
    # create and normalize the colormap
    cmap_colorbar = clr.LinearSegmentedColormap.from_list('azimuth_std_colorbar', colors, n_bins)
    norm_colorbar = clr.Normalize(0, 40)
    
    # get the colormaps for the plot (include white for the borders and black for the background)
    colors = [[0, 0, 0], [0, 0, 0.5], [0, 0, 1], 
              [0, 0.5, 1], [0, 1, 1], 
              [0.5, 1, 0.5], [1, 1, 0], 
              [1, 0.5, 0], [1, 0, 0], 
              [0.5, 0, 0], [1, 1, 1]]
    # create and normalize the colormap
    cmap_plt = clr.LinearSegmentedColormap.from_list('azimuth_std_plt', colors, n_bins)    
    
    _, ax = plt.subplots(figsize = (15,10))
    
    # get the borders between WM and GM
    edges = get_edges(mask, healthy = healthy)
    
    # change the values of the azimuth std to visualize the borders and remove the background
    for idx, x in enumerate(mask):
        for idy, y in enumerate(x):
            if healthy:
                if y == 0:
                    azimuth_std[idx, idy] = -5
            else:
                if sum(y) == 0:
                    azimuth_std[idx, idy] = -5
    
    for idx, x in enumerate(edges):
        for idy, y in enumerate(x):
            if y != 0:
                azimuth_std[idx, idy] = 45
        
    # plot the azimuth std
    ax.imshow(azimuth_std, cmap = cmap_plt)
    ax = plt.gca()

    # format the color bar
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm_colorbar, cmap=cmap_colorbar), pad = 0.02, 
                        ticks=np.arange(0, 41, 10), fraction=0.06)
    cbar.ax.set_yticklabels(["{:.0f}".format(a) for a in np.arange(0, 41, 10)], 
                            fontsize=40, weight='bold')

    # add a title and save the plot
    title = 'Azimuth of the optical axis'
    plt.title(title, fontsize=35, fontweight="bold", pad=14)
    plt.xticks([])
    plt.yticks([])
    try:
        os.remove(os.path.join(folder, 'annotation', 'azimuth_noise.pdf'))
    except:
        pass
    
    plt.savefig(os.path.join(folder, 'histology', 'azimuth_noise.pdf'))
    
    
def get_edges(mask: np.array, healthy: bool = False):
    """
    get_edges finds the border between the WM and the GM

    Parameters
    ----------
    mask : np.array
        the mask of the GM / WM
    healthy : bool
        whether the measurement is healthy tissue or not

    Returns
    -------
    img_dilation : np.array
        the dilated edges
    """
    if healthy:
        edges = cv2.Canny(mask, threshold1=30, threshold2=100)
    else:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # Taking a matrix of size 5 as the kernel and dilate the border
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)

    return img_dilation


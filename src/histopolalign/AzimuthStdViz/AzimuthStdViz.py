import os
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import circstd
from tqdm import tqdm
from matplotlib import cm, colors
import matplotlib.cbook
import matplotlib.colors as clr
import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import cv2


def get_and_plots_stds(measurements: list, sq_size: int = 4):
    azimuth_stds = {}

    for folder in tqdm(measurements):
        path_polarimetry = os.path.join(folder, 'polarimetry', '550nm')
        azimuth = np.load(os.path.join(path_polarimetry, 'MM.npz'))['azimuth']
        
        azimuth_std = np.zeros((round(azimuth.shape[0]), round(azimuth.shape[1])))
        for idx in range(0, len(azimuth), 1):
            for idy in range(0, len(azimuth[0]), 1):
                try:
                    assert azimuth[idx-sq_size//2:idx+sq_size//2,idy-sq_size//2:idy+sq_size//2].shape == ((4,4))
                    std = circstd(azimuth[idx-sq_size//2:idx+sq_size//2,idy-sq_size//2:idy+sq_size//2], high=180)
                    try:
                        azimuth_std[idx, idy] = std
                    except:
                        pass
                except:
                    pass
        
        azimuth_std[azimuth_std > 40] = 40
        azimuth_stds[folder] = azimuth_std
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


def get_edges(mask, healthy = False):
    if healthy:
        edges = cv2.Canny(mask, threshold1=30, threshold2=100)
    else:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)

    return img_dilation

def plot_azimuth_noise(azimuth_std, folder, mask, healthy = False):
    
    n_bins = 200
    colors = [[0, 0, 0.5], [0, 0, 1], 
              [0, 0.5, 1], [0, 1, 1], 
              [0.5, 1, 0.5], [1, 1, 0], 
              [1, 0.5, 0], [1, 0, 0], 
              [0.5, 0, 0]]
    # create and normalize the colormap
    cmap_colorbar = clr.LinearSegmentedColormap.from_list('azimuth_std_colorbar', colors, n_bins)
    norm_colorbar = clr.Normalize(0, 40)
    
    colors = [[0, 0, 0], [0, 0, 0.5], [0, 0, 1], 
              [0, 0.5, 1], [0, 1, 1], 
              [0.5, 1, 0.5], [1, 1, 0], 
              [1, 0.5, 0], [1, 0, 0], 
              [0.5, 0, 0], [1, 1, 1]]
    # create and normalize the colormap
    cmap_plt = clr.LinearSegmentedColormap.from_list('azimuth_std_plt', colors, n_bins)
    norm_plt = clr.Normalize(-5, 45)
    

    fig, ax = plt.subplots(figsize = (15,10))
    edges = get_edges(mask, healthy = healthy)
    
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
        
    
    
    ax.imshow(azimuth_std, cmap = cmap_plt)
    ax = plt.gca()

    # format the color bar
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm_colorbar, cmap=cmap_colorbar), pad = 0.02, 
                        ticks=np.arange(0, 41, 10), fraction=0.06)
    cbar.ax.set_yticklabels(["{:.0f}".format(a) for a in np.arange(0, 41, 10)], 
                            fontsize=40, weight='bold')


    title = 'Azimuth of the optical axis'
    plt.title(title, fontsize=35, fontweight="bold", pad=14)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(folder, 'annotation', 'azimuth_noise.pdf'))
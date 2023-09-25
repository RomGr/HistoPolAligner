"""
Based on the work from 'Image deformation using moving least squares'

@author: Jian-Wei ZHANG
@email: zjw.cs@zju.edu.cn
@date: 2017/8/8
@update: 2020/9/25
@update: 2021/7/14: Simplify usage
@update: 2021/12/24: Fix bugs and add an example of random control points (see `demo2()`)
"""

import numpy as np
import math
from skimage import measure, io
from PIL import Image
import cv2
import os
from histopolalign.MLS.img_utils import (
    mls_similarity_deformation, mls_rigid_deformation
)
import histopolalign.SkeletonMatching.skeletonContext as sc
import numpy as np
from scipy import ndimage
import time

from histopolalign.prepare_images import FolderAlignHistology
from histopolalign.helpers import load_param_matching_pts
import matplotlib.pyplot as plt
import random
random.seed(42)


def match_skeletons(measurement: FolderAlignHistology, border_parameter: int, nsamples: int = 150, max_distance: int = 100, Verbose: bool = False):
    """
    match_skeletons is the master function running a pipeline to match the skeletons of the two images (histology and polarimetry) and save the results

    Parameters
    ----------
    measurement: FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement 
    border_parameter : int
        the number of pixels to remove from the border of the image
    nsamples : int
        the number of samples to use for the skeleton matching (default is 150)
    max_distance : int
        the maximum distance between two points to be considered as a match (default is 100)
    Verbose : bool
        if True, print the time taken by the function (default is False)
    """
    start = time.time()
    path_histology_polarimetry_aligned = os.path.join(measurement.path_histology_polarimetry, 'aligned')
    
    # create the folder histology/aligned to store the outputs of the step
    try:
        os.mkdir(path_histology_polarimetry_aligned)
    except FileExistsError:
        pass
    
    # get the contour points and sample nsamples points from them
    histology_pts, intensity_pts = get_contour_points(measurement, border_parameter, path_histology_polarimetry_aligned)
    histology_pts, intensity_pts, nsamp = sample_points(histology_pts, intensity_pts, nsamples = nsamples)
    
    # determine the matching points based on the skeleton matching algorithm
    X2b, Y2, X2 = compute_matching_points(histology_pts['PointsList'], intensity_pts['PointsList'], 
                        histology_pts['t'], intensity_pts['t'], nsamp)
    matched_points = generate_matched_points(X2b, Y2, max_distance)
    
    # plot the matched points
    plot_matched_points(matched_points, path_histology_polarimetry_aligned, X2, Y2)

    # apply the MLS algorithm to the matched points
    imgs = warp_img(matched_points, [np.asarray(measurement.registration_img), np.asarray(measurement.registration_labels_img),
                                    np.asarray(measurement.registration_labels_GM_WM_img)])
    [measurement.histology_contour, measurement.labels_contour, measurement.labels_GM_WM_contour] = imgs
    link_alignement_folder = measurement.to_align_link

    # plot the contours for both images
    plot_contour(histology_pts['s'], path_histology_polarimetry_aligned, 'contour_histology.png')
    plot_contour(intensity_pts['s'], path_histology_polarimetry_aligned, 'contour_polarimetry.png')
    
    Image.fromarray(measurement.labels_GM_WM_contour).save(os.path.join(link_alignement_folder, 'histology_labels_GM_WM_upscaled.png'))
    Image.fromarray(measurement.labels_GM_WM_contour).save(os.path.join(path_histology_polarimetry_aligned, 'histology_labels_GM_WM_contour.png'))

    Image.fromarray(measurement.labels_contour).save(os.path.join(link_alignement_folder, 'histology_labels_upscaled.png'))
    Image.fromarray(measurement.labels_contour).save(os.path.join(path_histology_polarimetry_aligned, 'histology_labels_contour.png'))

    Image.fromarray(measurement.histology_contour).save(os.path.join(link_alignement_folder, 'histology_rgb_upscaled.png'))
    Image.fromarray(measurement.histology_contour).save(os.path.join(path_histology_polarimetry_aligned, 'histology_rgb_contour.png'))
    
    end = time.time()
    if Verbose:
        print("Match the two skeletons: {:.3f} seconds.".format(end - start))
    
def get_contour_points(measurement: FolderAlignHistology, border_parameter: int, path_histology_polarimetry_aligned: str):
    """
    function used to get the contours of the histology image and the intensity image, get the contour points lists and save the contours

    Parameters
    ----------
    measurement : FolderAlignHistology
        the FolderAlignHistology containing the information and the images about the measurement
    border_parameter : int
        the number of pixels to remove from the border of the image
    path_histology_polarimetry_aligned : str
        the path to the folder where the results will be saved

    Returns
    -------
    histology_pts : tuple
        the contour points of the histology image
    intensity_pts : tuple
        the contour points of the intensity image
    """
    # mask the histology image (i.e. remove the background and get the contour of the image)
    histology = measurement.registration_img.convert('L')
    
    # create the mask of the histology image and get the points of the contour of the histology image
    histology_mask, contour_histology = remove_border_mask(histology, treshold_min = 0, threshold_max = 256, kernel_nb = 10, border_parameter = border_parameter)
    histology_mask_img = Image.fromarray(histology_mask.astype(np.uint8) * 255)
    histology_mask_img.save(os.path.join(path_histology_polarimetry_aligned, 'histology_mask.png'))
    histology_pts = get_points_list(contour_histology)
        
    # get the points of the contour of the polarimetric intensity image
    intensity_mask = np.array(Image.open(measurement.annotation_path))
    intensity_mask, contour_filtered = remove_border_mask(intensity_mask, border_parameter = border_parameter, intensity = True)
    intensity_mask_img = Image.fromarray(intensity_mask.astype(np.uint8) * 255)
    intensity_mask_img.save(os.path.join(path_histology_polarimetry_aligned, 'intensity_mask.png'))
    intensity_pts = get_points_list(contour_filtered)
    return histology_pts, intensity_pts
    
    
def remove_border_mask(image: Image, treshold_min: int = 0, threshold_max: int = 200, kernel_nb: int = 10, border_parameter: int = 5, intensity: bool = False):
    """
    remove_border_mask allows to get the mask and the contour of an image

    Parameters
    ----------
    image : Pillow image
        the image to get the mask and the contour from
    treshold_min, threshold_max : int, int
        the tresholds to use to get the mask (default is 0 and 200)
    kernel_nb : int
        the size of the kernel to use to get the mask (default is 12)

    Returns
    -------
    intensity_mask : Pillow image
        the mask of the image (i.e. the background is removed)
    contour_filtered : Pillow image
        the contour of the image
    """
    # get the mask and fill the holes in the images
    if not intensity:
        mask = get_mask_and_contour(image, treshold_min = treshold_min, threshold_max = threshold_max, kernel_nb = kernel_nb)
        mask = flood_fill(mask)
    else:
        mask = image
    
    # remove the border of the image
    output_corrected = np.zeros(mask.shape)
    for idx, x in enumerate(mask):
        for idy, y in enumerate(x):
            if idx < border_parameter or idx > mask.shape[0] - border_parameter or idy < border_parameter or idy > mask.shape[1] - border_parameter:
                pass
            else:
                output_corrected[idx, idy] = y

    # find the contours of the ROI
    c = measure.find_contours(output_corrected.astype(np.uint8), fully_connected = 'high')
    contoured = np.zeros(mask.shape)
    for cont in c:
        for pt in cont:
            contoured[math.ceil(pt[0]), math.ceil(pt[1])] = 1
            
    return output_corrected, contoured
    
    
def get_mask_and_contour(intensity: Image, treshold_min: int, threshold_max: int, kernel_nb = 12):
    """
    get_mask_and_contour allows to get the mask and the contour of an image

    Parameters
    ----------
    intensity : Pillow image
        the image to get the mask and the contour from
    treshold_min, threshold_max : int
        the tresholds to use to get the mask (default is 0 and 200)
    kernel_nb : int
        the size of the kernel to use to get the mask (default is 12)

    Returns
    -------
    output : Pillow image
        the mask of the image (i.e. the background is removed)
    """
    # remove the background
    output = np.logical_and(np.array(intensity) > treshold_min, np.array(intensity) < threshold_max)
    
    # Creating kernel
    kernel = np.ones((kernel_nb, kernel_nb), np.uint8)
    output = cv2.erode(output.astype(np.uint8), kernel) 
    output = cv2.dilate(output.astype(np.uint8), kernel)
    return output


def flood_fill(array: np.array):
    """
    flood_fill allows to fill the holes in the histology image, obtained from http://arcgisandpython.blogspot.com/2012/01/python-flood-fill-algorithm.html
    
    Parameters
    ----------
    array : np.array
        the array to fill the holes from
    
    Returns
    -------
    output_array : np.array
        the array with the holes filled
    """
    input_array = np.copy(array)
    # set h_max to a value larger than the array maximum to ensure that the while loop will terminate
    h_max = np.max(input_array * 2.0)
    # build mask of cells with data not on the edge of the image, use 3x3 square structuring element
    # build Structuring element only using NumPy module, structuring element could also be built using SciPy ndimage module
    data_mask = np.isfinite(input_array)
    inside_mask = ndimage.binary_erosion(data_mask, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool))
    
    # initialize output array as max value test_array except edges
    output_array = np.copy(input_array)
    output_array[inside_mask] = h_max

    # array for storing previous iteration
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   

    # iterate until marker array doesn't change
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(
            input_array,
            ndimage.grey_erosion(output_array, size=(3, 3)))
        
    return output_array


def get_points_list(img: np.array):
    """
    get_points_list allows to get the points located on the contour of an image
    
    Parameters
    ----------
    img : np.array
        the array to get the points from
    
    Returns
    -------
    results : dict
        the dictionary containing the image, the tangent, the points list and the detected edge list
    """
    img = img.astype('float64')
    edglelists = []
    edglelists_len = []
    
    # fint the edge list starting from a random point
    for j in range(5):
        edge_list = find_edge_list(img)
        edglelists.append(edge_list[0])
        edglelists_len.append(edge_list[1])
    
    # get the tangent estimate and the points list
    Tangent, PointsList = skeletonTangentEstimate([edglelists[np.argmax(edglelists_len)]])
    results = {'s': img, 'Tangent': Tangent, 'PointsList': PointsList, 'edgeList': edglelists[np.argmax(edglelists_len)]}
    return results


def find_edge_list(img):
    """
    find_edge_list allows to get the list of the points located on the contour of an image
    
    Parameters
    ----------
    img : np.array
        the array to get the contour list from
    
    Returns
    -------
    edge_list : list
        edge_list is the list of the points located on the contour of the image
    len(edge_list) : int
        the length of the edge_list
    """
    edge_list = []
    found = False
    
    # find a random point on the contour
    while not found:
        x = random.randrange(img.shape[0])
        y = random.randrange(img.shape[1])

        if img[x, y] == 1:
            start = [x,y]
            edge_list.append([x,y])
            found = True

    # initialize the variables
    finished = False
    current_x = start[0]
    current_y = start[1]
    counter = 0

    # find the edge list while the algorithm is not finished and the counter is not too high
    while not finished and counter < 100:
        # find the neighbors of the current point
        neighbors = img[current_x - 1: current_x + 2, current_y - 1: current_y + 2]
        
        # get the next point
        found = False
        for idx, x in enumerate(neighbors):
            for idy, y in enumerate(x):
                if y == 1 and not found:
                    if [current_x - 1 + idx , current_y - 1 + idy] in edge_list:
                        pass
                    else:
                        edge_list.append([current_x - 1 + idx , current_y - 1 + idy])
                        current_x = current_x - 1 + idx
                        current_y = current_y - 1 + idy
                        found = True
                       
        # enlarge the neighbors if the point is not found 
        if not found:
            neighbors_enlarged = img[current_x - 4: current_x + 5, current_y - 4: current_y + 5]
            for idx, x in enumerate(neighbors_enlarged):
                for idy, y in enumerate(x):
                    if y == 1 and not found:
                        if [current_x - 4 + idx , current_y - 4 + idy] in edge_list:
                            pass
                        else:
                            edge_list.append([current_x - 4 + idx , current_y - 4 + idy])
                            current_x = current_x - 4 + idx
                            current_y = current_y - 4 + idy
                            found = True
        if not found:
            finished = True
    return edge_list, len(edge_list)


def skeletonTangentEstimate(edgeList: list, landMarkPoints: int = 4):
    """
    skeletonTangentEstimate allows to get the tangent estimate and the points list
    
    Parameters
    ----------
    edgeList : list
        the list of the branch points
    landMarkPoints : int
        the number of points used to estimate the tangent
    
    Returns
    -------
    estimatedTangent : np.array
        the estimated tangent
    skeletonPointsList : list
        the list of the points located on the contour of the image
    """
    estimatedTangent = np.array([])
    skeletonPointsList = []
    for branchPoints in edgeList:
        numPoints = len(branchPoints)
        skeletonPointsList.extend(branchPoints)
        estimatedTangentBranch = np.zeros(numPoints)
        for i, _ in enumerate(branchPoints):
            if i <= landMarkPoints:
                startPoint = np.array(branchPoints[0])
            else:
                startPoint = np.array(branchPoints[i - landMarkPoints])
            if i >= numPoints - landMarkPoints:
                endPoint = np.array(branchPoints[-1])
            else:
                endPoint = np.array(branchPoints[i + landMarkPoints])
            vector = endPoint - startPoint
            tan = vector[0] / vector[1] if vector[1] != 0 else np.inf
            estimatedTangentBranch[i] = np.arctan(tan)
        estimatedTangent = np.concatenate((estimatedTangent, estimatedTangentBranch))
    return estimatedTangent, skeletonPointsList


def sample_points(histology_pts: dict, intensity_pts: dict, nsamples: int):
    """
    sample_points allows to sample nsamples points from the contours of the histology and the intensity images
    
    Parameters
    ----------
    histology_pts : dict
        the dictionary containing the image, the tangent, the points list and the detected edge list for the histology image
    intensity_pts : dict
        the dictionary containing the image, the tangent, the points list and the detected edge list for the intensity image
    nsamples : int
        the number of points to sample
    
    Returns
    -------
    histology_pts : dict
        the dictionary containing the image, the tangent, the points list and the detected edge list for the histology image with the sampled points
    intensity_pts : dict
        the dictionary containing the image, the tangent, the points list and the detected edge list for the intensity image with the sampled points
    """
    # sample some points from both contours
    nsamp = min(len(histology_pts['PointsList']) - 1, len(intensity_pts['PointsList']) - 1, nsamples)
    
    histology_pts['t'] = sc.bdry_extract(histology_pts['s'], histology_pts['PointsList'])
    histology_pts['PointsList'], histology_pts['t'], histology_pts['tTangent'] = sc.get_samples(histology_pts['PointsList'], histology_pts['t'], 
                                                                                               histology_pts['Tangent'], nsamp)  
    
    intensity_pts['t'] = sc.bdry_extract(intensity_pts['s'], intensity_pts['PointsList'])
    intensity_pts['PointsList'], intensity_pts['t'], intensity_pts['tangeant'] = sc.get_samples(intensity_pts['PointsList'], intensity_pts['t'], 
                                                                                               intensity_pts['Tangent'], nsamp)
    
    return histology_pts, intensity_pts, nsamp
    

def compute_matching_points(s1PointsList, s2PointsList, t1, t2, nsamp):
    """
    compute_matching_points computes the matching points between two skeletons using the skeleton context algorithm
    
    Parameters
    ----------
    s1PointsList : np.array
        the array of points in the first skeleton
    s2PointsList : np.array
        the array of points in the second skeleton
    t1 : np.array
        the array of tangent vectors in the first skeleton
    t2 : np.array
        the array of tangent vectors in the second skeleton
    nsamp : int
        the number of points in the skeletons
    
    Returns
    -------
    X2b : np.array
    X2 : np.array
        the array of matched points in the first skeleton
    Y2 : np.array
        the array of matched points in the second skeleton
    """
    # set up all the parameters
    parameters = load_param_matching_pts()
    X = np.copy(s1PointsList)
    Y = np.copy(s2PointsList)
    Xk = X
    tk = t1
    numDumPoints = int(parameters['numDumRate'] * nsamp)
    outVec1 = np.zeros(nsamp)
    outVec2 = np.zeros(nsamp)
    neighborMap = np.tile(np.arange(1,nsamp+1),(nsamp,1))
    neighborCost = np.zeros((nsamp,nsamp))
    
    # loop and repeat the skeleton context algorithm for numIter times
    for k in range(parameters['numIter']):
        pointHistogram1, _ = sc.skeletonContext(Xk.T,np.zeros(nsamp),parameters['nbins_theta'],parameters['nbins_r'],
                                                            parameters['r_inner'],parameters['r_outer'],outVec1)
        pointHistogram2, meanDistance2 = sc.skeletonContext(Y.T,np.zeros(nsamp),parameters['nbins_theta'],parameters['nbins_r'],
                                                            parameters['r_inner'],parameters['r_outer'],outVec2)
        
        if parameters['affineStartFlag']:
            if k == 0:
                lambda_o = 1000
            else:
                lambda_o = parameters['betaInit'] * pow(parameters['r'],k-1)
        else:
            lambda_o = parameters['betaInit'] * pow(parameters['r'],k)
        beta_k=(pow(meanDistance2,2)) * lambda_o

        costMatShape = sc.HistCost(pointHistogram1, pointHistogram2)
        thetaDiff = np.tile(tk,[nsamp,1]).T - np.tile(t2,[nsamp,1])
        
        if parameters['polarityFlag']:
            costMatTheta = 0.5 * (1 - np.cos(thetaDiff))
        else:
            costMatTheta = 0.5 * (1 - np.cos(2 * thetaDiff))

        costMat = (1 - parameters['thetaWeight']) * costMatShape + parameters['thetaWeight'] * costMatTheta
        costMat += neighborCost

        # Calculate Skeleton Context cost
        costMatTemp = costMat - neighborCost
        a1 = np.min(costMatTemp,0)
        a2 = np.min(costMatTemp,1)
        parameters['skeletonMatchCost'][k] = max(np.mean(a1),np.mean(a2))
        
        numMatchPoints = nsamp + numDumPoints
        costMatDum = parameters['costDum'] * np.ones((numMatchPoints,numMatchPoints))
        costMatDum[0:nsamp,0:nsamp] = costMat
        matchedVec, _ = sc.hungarian(costMatDum)
        matchedVec2 = np.argsort(matchedVec)
        
        '''Neighboring Effect'''
        s1PointsMatchedInd = matchedVec2[:nsamp] <= nsamp
        s1PointsMatched = np.argwhere(s1PointsMatchedInd)
        matchDifference = s1PointsMatched - matchedVec2[s1PointsMatched]
        matchDifferenceMean = np.mean(matchDifference)
        matchDifferenceStd = np.std(matchDifference)
        outlierInd = (matchDifference >=  (matchDifferenceMean + 2.5 * matchDifferenceStd)) | (matchDifference <=  (matchDifferenceMean - 2.5 * matchDifferenceStd))
        parameters['matchRatio'][k] = len(s1PointsMatched) / nsamp
        if parameters['matchRatio'][k] <= 0.65:
            if (k != 0 & outlierInd.any() ):
                outlierMatchS2Points = s1PointsMatched[outlierInd]
                outlierMatchS1Points = matchedVec2[outlierMatchS2Points]
                matchedVec2[outlierMatchS2Points] = numMatchPoints
                matchedVec[outlierMatchS1Points] = numMatchPoints

                s1PointsMatchedInd = matchedVec2[:nsamp] <= nsamp
                s1PointsMatched = np.argwhere(s1PointsMatchedInd)
                matchDifference = s1PointsMatched - matchedVec2[s1PointsMatched]
                matchDifferenceMean = np.mean(matchDifference)
                matchDifferenceStd = np.std(matchDifference)

            neighborMean = neighborMap.T - matchDifferenceMean
            neighborCost = parameters['neighborWeight'] * (1 - np.exp(- (neighborMap - neighborMean)**2 / (2 * 10**2)))
        else:
            neighborCost = np.zeros((nsamp,nsamp))

        outVec1 = matchedVec2[:nsamp] > nsamp
        outVec2 = matchedVec[:nsamp] > nsamp
    
        X2 = np.nan * np.ones((numMatchPoints,2))
        X2[:nsamp,:] = np.copy(Xk)
        X2 = X2[matchedVec - 1,:]
        X2b = np.nan * np.ones((numMatchPoints,2))
        X2b[:nsamp,:] = X
        X2b = X2b[matchedVec - 1,:]
        Y2 = np.nan * np.ones((numMatchPoints,2))
        Y2[:nsamp,:] = np.copy(Y)
        
        indGood = np.where(~np.isnan(X2b[:nsamp,0]))[0]
        numGood = len(indGood)
        X3b = X2b[indGood,:]
        Y3  = Y2[indGood,:]
        
        # Calculate bending energy
        cx,cy,E,_ = sc.bookstien(X3b , Y3, beta_k)
        parameters['bendingEnergy'][k] = E
        # Calculating affine cost
        A = np.vstack((cx[numGood + 1:numGood + 3], cy[numGood + 1:numGood + 3]))
        _,s,_ = np.linalg.svd(A)
        parameters['affineCost'][k] = np.log(s[0]/s[1])

        # warp coordinates
        fx_aff = np.dot(cx[numGood:numGood + 3].T,np.vstack((np.ones((1,nsamp)),X.T)))
        d2 = sc.dist2(X3b, X)
        U = d2 * np.log(d2 + np.finfo(float).eps)
        fx_wrp = np.dot(cx[:numGood].T,U)
        fx = fx_aff + fx_wrp
        fy_aff = np.dot(cy[numGood:numGood + 3].T,np.vstack((np.ones((1,nsamp)),X.T)))
        fy_wrp = np.dot(cy[:numGood].T,U)
        fy = fy_aff + fy_wrp
        
        Z=np.vstack((fx,fy)).T
        Xtan = X + parameters['tan_eps'] * np.vstack((np.cos(t1),np.sin(t1))).T
        fx_aff = np.dot(cx[numGood:numGood + 3].T,np.vstack((np.ones((1,nsamp)),Xtan.T)))
        d2 = sc.dist2(X3b, Xtan)
        U = d2 * np.log(d2 + np.finfo(float).eps)
        fx_wrp = np.dot(cx[:numGood].T,U)
        fx = fx_aff + fx_wrp
        fy_aff = np.dot(cy[numGood:numGood + 3].T,np.vstack((np.ones((1,nsamp)),Xtan.T)))
        fy_wrp = np.dot(cy[:numGood].T,U)
        fy = fy_aff + fy_wrp

        Ztan = np.vstack((fx,fy)).T
        tk = np.arctan2(Ztan[:,1] - Z[:,1], Ztan[:,0] - Z[:,0])
        
        # Update Paramters
        Xk = Z
        matchingData = np.vstack((parameters['bendingEnergy'][0:k+1],parameters['affineCost'][0:k+1],
                                  parameters['skeletonMatchCost'][0:k+1],parameters['matchRatio'][0:k+1])).T
        if np.sum(matchingData[-1,0:3]) < 0.9:
            break
    
    return X2b, Y2, X2


def generate_matched_points(X2b: np.array, Y2: np.array, dist: int):
    """
    generate_matched_points allows to remove the values for which the distance is bigger than dist
    
    Parameters
    ----------
    X2b : np.array
        the points of the first skeletons   
    Y2 : np.array
        the corresponding points of the second skeleton
    dist : int
        the distance threshold to be considered as a match
    
    Returns
    -------
    matched_points : dict
        the matched points between the two skeletons with a distance smaller than dist
    """
    matched_points = {}
    Xs = []
    Ys = []
    for x, y in zip(X2b, Y2):
        if not(math.isnan(x[0]) or math.isnan(x[1]) or math.isnan(y[0]) or math.isnan(y[1])):
            dist = math.sqrt( (x[1] - y[1])**2 + (x[0] - y[0])**2 )
            if dist > 100:
                pass
            else:
                matched_points[tuple(x)] = y
                Xs.append(x)
                Ys.append(y)
    return matched_points


def warp_img(matched_points: dict, images: list):
    """
    warp_img uses the matched points to warp the second image to the first one
    
    Parameters
    ----------
    matched_points : dict
        the matched points between the two skeletons with a distance smaller than dist  
    images : list
        the two images to be warped
    dist : int
        the distance threshold to be considered as a match
    
    Returns
    -------
    aug2s : list
        the warped images
    """
    # get the list of matched points
    p = np.array([list(sr) for sr in list(matched_points.keys())])
    q = np.array([list(sr) for sr in list(matched_points.values())])
    
    # set the grid
    height, width, _ = images[0].shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    # compute the similarity matrix
    similar = mls_rigid_deformation(vy, vx, p, q, alpha=1)
    
    # apply the similarity matrix to the second image
    aug2s = []
    for image in images:
        aug2 = np.ones_like(image)
        aug2[vx, vy] = image[tuple(similar)]
        aug2s.append(aug2)
    return aug2s


def plot_matched_points(matched_points, path_histology_polarimetry_aligned, X2, Y2):
    """
    plot_matched_points is the function used to plot the matched points between the two skeletons
    
    Parameters
    ----------
    matched_points : dict
        the matched points between the two skeletons with a distance smaller than dist  
    path_histology_polarimetry_aligned : str
        the path to the folder where the images will be saved
    X2 : np.array
        the points of the first skeletons
    Y2 : np.array
        the corresponding points of the second skeleton
    """
    # get the matched points
    xs = [list(sr) for sr in matched_points.keys()]
    ys = [list(sr) for sr in matched_points.values()]
    xs_st = []
    ys_st = []
    for x, y in zip(xs, ys):
        ys_st.append(x[0])
        ys_st.append(y[0])

        xs_st.append(x[1])
        xs_st.append(y[1])

    # reorganize the points in the correct format
    xx = np.vstack([xs_st[0::2],xs_st[1::2]])
    yy = np.vstack([ys_st[0::2],ys_st[1::2]])

    # plot the matched points
    fig, ax = plt.subplots()
    ax.scatter(X2[:,1],X2[:,0],c='b', marker='+',s=20)
    ax.scatter(Y2[:,1],Y2[:,0],c='r',marker='o',s=20)
    _ = ax.plot(xx,yy)
    ax.invert_yaxis()
    fig.patch.set_visible(False)
    ax.axis('off')

    # and save the figure
    plt.savefig(os.path.join(path_histology_polarimetry_aligned, 'matched_points.png')) 
    plt.savefig(os.path.join(path_histology_polarimetry_aligned, 'matched_points.pdf')) 
    plt.close()


def plot_contour(img, path_histology_polarimetry_aligned, strpath):
    """
    plot_contour is the function used to plot the contour of the image
    
    Parameters
    ----------
    s1 : np.array
        the first image with the contour
    edgelist : list
        the list of points of the contour
    path_histology_polarimetry_aligned : str
        the path to the folder where the images will be saved
    strpath : str
        the name of the image to be saved
    """
    # taking a matrix of size 5 as the kernel to dilate the contour
    kernel = np.ones((7, 7), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    img = Image.fromarray(img_dilation.astype(np.uint8)*255)
    img.save(os.path.join(path_histology_polarimetry_aligned, strpath))
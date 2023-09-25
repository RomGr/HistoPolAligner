def get_mask_border(img_labels_propagated, color_code_link, distance = 10, val: int = 10):
    """
    is used to remove the border of the image (i.e. remove the possible border effect for the sample)

    Parameters
    ----------
    img_labels_propagated : Pillow Image
        the final labels image
    distance : int
        the distance to remove (default = 30)

    Returns
    -------
    mask_border : np.array
        an array containing the pixels to keep and to remove
    """
    img_labels_propagated_reverse = np.zeros(img_labels_propagated.shape[0:2])
    for idx, x in enumerate(img_labels_propagated):
        for idy, y in enumerate(x):
            if color_code_link[tuple(y)] == 7:
                img_labels_propagated_reverse[idx, idy] = 1
            else:
                img_labels_propagated_reverse[idx, idy] = 0
                
    zero = cv2.findNonZero(img_labels_propagated_reverse)
    
    # precompute the distances
    distances_1st_axis = {}
    for idx in range(img_labels_propagated_reverse.shape[1]):
        distances_1st_axis[idx] = (zero[:,:,0] - idx) **2
    distances_2nd_axis = {}
    for idx in range(img_labels_propagated_reverse.shape[1]):
        distances_2nd_axis[idx] = (zero[:,:,1] - idx) **2
        
    
    mask_border = np.zeros(img_labels_propagated.shape[0:2])
    for idx, x in enumerate(img_labels_propagated):
        for idy, y in enumerate(x):

            if color_code_link[tuple(y)] == 7:
                pass
            else:
                target = (idy, idx)
                res = find_nearest_white(zero, target, distances_1st_axis, distances_2nd_axis, bd = True)
                idx_min, idy_min = res[0][1], res[0][0]
                dist = np.linalg.norm(np.array([idx, idy]) - np.array((idx_min, idy_min)))
                if dist < distance:
                    pass
                else:
                    mask_border[idx, idy] = 1
                
    return mask_border
from histopolalign import align_imgs
import imagej
import pickle
import os

def main():
    """
    main function to perform the alignment of the histology images with the polarimetry images

    Parameters
    ----------
    
    Returns
    -------
    """
    # load the alignment measurements object
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)).split('src')[0], 'notebooks', 'temp', 'align_measurement.pickle'), 'rb') as handle:
        alignment_measurements = pickle.load(handle)
        
    # initialize the imageJ session
    ij = imagej.init(os.path.join(os.path.dirname(os.path.realpath(__file__)).split('src')[0], 'Fiji.app'), mode='interactive')
    ij.ui().showUI() # if you want to display the GUI immediately
    
    # align the images
    align_imgs.align_w_imageJ(ij, alignment_measurements)
    
    # save the alignment measurements object
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)).split('src')[0], 'notebooks', 'temp', 'align_measurement.pickle'), 'wb') as handle:
        pickle.dump(alignment_measurements, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    main()
    

from histopolalign.AlignImages import align_imgs
import imagej
import pickle
import os
import sys
path_home = os.path.join(os.path.dirname(os.path.realpath(__file__)).split('src')[0], 'Fiji.app', 'java')
assert len(os.listdir(path_home)) == 1
path_home = os.path.join(path_home, os.listdir(path_home)[0])
assert len(os.listdir(path_home)) == 1
path_home = os.path.join(path_home, os.listdir(path_home)[0])
assert len(os.listdir(path_home)) == 1
path_home = os.path.join(path_home, os.listdir(path_home)[0])
os.environ['JAVA_HOME'] = path_home
    

def main():
    """
    main function to perform the alignment of the histology images with the polarimetry images

    Parameters
    ----------
    
    Returns
    -------
    """
    print('here')
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
    

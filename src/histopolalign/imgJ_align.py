from histopolalign import align_imgs
import imagej
import pickle
import os

def main():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)).split('src')[0], 'notebooks', 'temp', 'align_measurement.pickle'), 'rb') as handle:
        alignment_measurements = pickle.load(handle)
        

    ij = imagej.init(r'C:\Users\romai\Downloads\fiji-win64\Fiji.app', mode='interactive')
    ij.ui().showUI() # if you want to display the GUI immediately
    align_imgs.align_w_imageJ(ij, alignment_measurements)
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)).split('src')[0], 'notebooks', 'temp', 'align_measurement.pickle'), 'wb') as handle:
        pickle.dump(alignment_measurements, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    main()
    

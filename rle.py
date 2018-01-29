import numpy as np
from glob import glob
import os
from skimage import util,io
import skimage

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prepare_submission():
    image_paths = glob('runs/1517125894.2133157/*')
    for image_path in image_paths :
        image = skimage.io.imread(image_path, as_grey=True)
        #gaussian blur after improving algo
        print(os.path.basename(os.path.splitext(image_path)[0]))
        print(rle_encoding(util.invert(image)))
        #write to a file

if __name__ == '__main__':
    prepare_submission()
import numpy as np
from glob import glob
import os
from skimage import util,io
import skimage
import csv
from skimage.morphology import label # label regions

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
    return " ".join(map(str,run_lengths))

def prob_to_rles(x, cut_off = 0.5):
    lab_img = label(x>cut_off)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)

def prepare_submission():
    image_paths = glob('runs/1517306309.1110826/*')
    f = open('stage1_final_submission.csv', 'wt')
    writer = csv.writer(f)
    writer.writerow(('ImageId', 'EncodedPixels'))
    for image_path in image_paths :
        image = skimage.io.imread(image_path, as_grey=True)
        #skimage.io.imsave("invert_"+image_path, util.invert(image))
        #gaussian blur after improving algo
        print(os.path.basename(os.path.splitext(image_path)[0]))
        #print(rle_encoding(util.invert(image)))

        rles = prob_to_rles(util.invert(image), cut_off = 0.5)
        rles_string = list(rles)
        rles_string =  " ".join(map(str,rles_string))
        writer.writerow( (os.path.basename(os.path.splitext(image_path)[0]), rles_string))
        #write to a file

if __name__ == '__main__':
    prepare_submission()
import numpy as np
from glob import glob
import os
from skimage import util,io
import skimage
from skimage.segmentation import clear_border
import csv
from skimage.morphology import label # label regions
from skimage.color import label2rgb

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, image_path):
    # remove artifacts connected to image border
    #cleared = clear_border(x)
    # label image regions
    lab_img = label(x)
    skimage.io.imsave("label_runs/"+image_path+".png",  label2rgb(lab_img, x))
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def prepare_submission():
    new_test_ids = []
    rles = []
    image_paths = glob('runs/1517306309.1110826/*')
    f = open('stage1_final_submission.csv', 'wt')
    writer = csv.writer(f)
    writer.writerow(('ImageId', 'EncodedPixels'))
    for image_path in image_paths :
        image = skimage.io.imread(image_path, as_grey=True)
        #skimage.io.imsave("invert_"+image_path, util.invert(image))
        #gaussian blur after improving algo
        #print(os.path.basename(os.path.splitext(image_path)[0]))
        #print(rle_encoding(util.invert(image)))
        id_ = os.path.basename(os.path.splitext(image_path)[0])
        rle = list(prob_to_rles(image, id_))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    for i in range(0, len(new_test_ids)):
        writer.writerow((new_test_ids[i], rle_to_string(rles[i])))

        #write to a file

if __name__ == '__main__':
    prepare_submission()
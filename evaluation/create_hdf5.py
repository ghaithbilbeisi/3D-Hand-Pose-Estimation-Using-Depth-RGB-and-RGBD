import h5py
import os
import cv2
import numpy as np
import util

##################################
# Preprocess images before running this script
# Remember to change dataset depending on type of data (fpad for depth, and fpac for rgb)
# Run for both phase='test' and phase='train' to get both datafiles
##################################

################ Confirm dataset before running this script! ###############################
dataset = 'fpad' # fpac or fpad

DIR = "../labels/"
phase = 'test' # test/train
h5_fn = os.path.join(DIR, (dataset + '_'+phase+'_data.h5'))


############## Dir containing preprocessed images ###############################
base_dir = '/home/bilbeisi/REN/cropped/'

names = util.load_names(dataset, phase)
labels = util.load_labels(dataset, phase)

if dataset == 'fpad':
    imgs = np.zeros( (len(names), 1, 96, 96), dtype=np.float32 ) # depth
else:
    imgs = np.zeros( (len(names), 3, 96, 96), dtype=np.float32 )
lbls = np.zeros( (len(labels), 63), dtype=np.float )

for idx, name in enumerate(names):
    if dataset == 'fpac':
        name = name.replace('.jpeg','.png')
    img = util.load_image(dataset, os.path.join(base_dir, name))
    img = img.astype(float)
    # revert back to normalized -1,1 since images where saved in 0,255 to allow viewing/verifying
    img[:] *= 2
    img[:] /= 255
    img[:] -= 1
    if dataset == 'fpac':
        # channels should be first axis for Caffe
        img = np.swapaxes(img, 0, 2)
    imgs[idx] = img
    lbls[idx] = np.asarray(labels[idx].split(), dtype=np.float32)

    if idx % 500 == 0:
        # print progress
        print('{}/{}'.format(idx + 1, len(names)))

print('finished reading ', len(names), ' images. Creating HDF5 file...')

# store image and label data in the hdf5 file
with h5py.File(h5_fn, 'w') as f:
   f['data'] = imgs

   f['labels'] = lbls

# create a text file containing hdf5 file names to pass as a network parameter
text_fn = os.path.join(DIR, (dataset + '_'+phase+'_data.txt'))
with open(text_fn, 'w') as f:
   print(h5_fn, file = f)

print('Done.')

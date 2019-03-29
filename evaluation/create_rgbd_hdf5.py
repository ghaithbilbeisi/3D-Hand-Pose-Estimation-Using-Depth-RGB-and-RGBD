import h5py
import os
import cv2
import numpy as np
import util
import time
import image_loops

##################################
# This script creates hdf5 files for RGB-D data since different processing is required
# Run for both phase='test' and phase='train' to get both datafiles
# colormap and depth2cords cython functions are used here
##################################

DIR = "../labels/"
phase = 'train'
h5_fn = os.path.join(DIR, ('rgbd_'+phase+'_data_.h5'))


base_dir = '/home/bilbeisi/REN/cropped/'

names = util.load_names('fpad', phase)
labels = util.load_labels('fpad', phase)
cnames = util.load_names('fpac', phase)
centers = util.load_centers('fpad', phase).astype(float)

imgs = np.zeros( (len(names), 4, 96, 96), dtype=np.float32 )
lbls = np.zeros( (len(labels), 63), dtype=np.float )

cube_size = 150     # cube size in mm for cropping

for idx, name in enumerate(names):
    cname = cnames[idx]
    img = util.load_image('fpad', os.path.join('/home/bilbeisi/REN/', name))
    img[img == 0] = 1

    cimg = util.load_image('fpac', os.path.join('/home/bilbeisi/REN/', cname))
    cimg = cimg.astype(float)


    cords_d = np.zeros((480,640,3))

    # using cython functions to drastically increase the speed of image pixel looping
    # loops over all depth image pixels and assigns (pixel coordinates, depth value) to cords_d
    cords_d = image_loops.depth2cords(img, cords_d)
    cords_d = np.reshape(cords_d, (-1,3))
    # from depth image pixels to world coordinates
    cords_3d = util.pixel2world(cords_d, 'fpad')
    # from world coordinates to rgb image coordinates, gives us an array containing the mapping of each depth image pixel to its corresponding rgb image pixel (if it exists)
    cords_c, skel_camcoords = util.world2pixel(cords_3d, 'fpac')
    cords_3d = np.reshape(cords_3d, (480,640,-1))

    img_rgbd = np.zeros((img.shape[0],img.shape[1],4))
    cords_c = np.reshape(cords_c, (480,640,3))

    # using cython functions to drastically increase the speed of image pixel looping
    # loops over all RGB-D image pixels and assigns RGB value using cimg and cords_c
    img_rgbd = image_loops.color_map(img, cimg, img_rgbd, cords_c)
    img_rgbd = np.asarray(img_rgbd)

    # use center to crop the image to required input size then normalize depth and RGB values
    center = centers[idx]
    crop = util.crop_image(img_rgbd, center, 'fpad')
    # norm depth values
    crop[:,:,3] -= center[2]
    crop[:,:,3] = np.maximum(crop[:,:,3], -cube_size)
    crop[:,:,3] = np.minimum(crop[:,:,3], cube_size)
    crop[:,:,3] /= cube_size
    # norm RGB values
    crop[:,:,:3] *= 2
    crop[:,:,:3] /= 255
    crop[:,:,:3] -= 1

    # swap axes; channels should be the first axis
    imgs[idx] = np.swapaxes(crop, 0, 2)
    lbls[idx] = np.asarray(labels[idx].split(), dtype=np.float32)
    # print progress
    if idx % 500 == 0:
        print('{}/{}'.format(idx + 1, len(names)))

print('finished processing ', len(names), ' images. Creating HDF5 file...')

# store image and label data in the hdf5 file
with h5py.File(h5_fn, 'w') as f:
   f['data'] = imgs

   f['labels'] = lbls

# create a text file containing hdf5 file names to pass as a network parameter
text_fn = os.path.join(DIR, ('rgbd_'+phase+'_data_.txt'))
with open(text_fn, 'w') as f:
   print(h5_fn, file = f)

print('Done.')

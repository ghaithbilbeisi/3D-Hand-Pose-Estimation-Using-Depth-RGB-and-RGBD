import cv2
import numpy as np
import math
import os
import sys
import image_loops
import util
from util import crop_image
from matplotlib import pyplot as plt
np.set_printoptions(threshold= np.nan)

########################
## This is the validation script for RGB-D. The creation and cropping of the RGBD images are done in craete_rgbd_hdf5.py because the "images" cannot be stored in the intermediate step between creation/cropping and moving to hdf5
## Some directories may need to be created before running some validation segments if they do not exist
## All preprocessing of the labels and centers is identical to that of depth therefore there is no need to redo it..
########################

dataset = 'rgbd'
phase = 'test'
root_dir = '/home/bilbeisi/REN/'



############################# Create RGB-D Images #################################
names = util.load_names('fpad', phase)
labels = util.load_labels('fpad', phase)
cnames = util.load_names('fpac', phase)
centers = util.load_centers('fpad', phase).astype(float)
imgs = np.zeros( (len(names), 4, 96, 96), dtype=np.float32 )
lbls = np.zeros( (len(labels), 63), dtype=np.float )

cube_size = 150     # cube size in mm for cropping

for idx, name in enumerate(names):
    if idx%1000 == 0:
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

        img /= 1160
        img *= 255

        # using cython functions to drastically increase the speed of image pixel looping
        # loops over all RGB-D image pixels and assigns RGB value using cimg and cords_c
        img_rgbd = image_loops.color_map(img, cimg, img_rgbd, cords_c)
        img_rgbd = np.asarray(img_rgbd)

        cv2.imwrite('/home/bilbeisi/REN/samples/rgbd/'+str(idx)+'.png', img_rgbd[:,:,:3])



############################# Draw predicted pose on RGB-D samples #################################
phase = 'test'
lbls = util.load_labels('fpad', phase) ### load test/train data
names = util.load_names('fpad', phase)
cnames = util.load_names('fpac', phase)
centers = util.load_centers('fpad', phase).astype(float)
lbls, preds = util.load_logs('fpad', 'rgbd_test_b159_lr_1e-2_xyz_1_200k_2_20k_.txt', centers)


for idx, name in enumerate(names):
    action = name.split('/')[2]
    subject = name.split('/')[1]
    if action == 'close_juice_bottle' and subject == 'Subject_2':
    # if idx in (11000, 13000, 15000, 16000, 18000, 21000, 22000, 23000, 31000):
        img = util.load_image('fpad', os.path.join(root_dir, name))
        img[img == 0] = 1

        cname = cnames[idx]
        cimg = util.load_image('fpac', os.path.join(root_dir, cname))
        cimg = cimg.astype(float)

        cords_d = np.zeros((480,640,3))
        cords_d = image_loops.depth2cords(img, cords_d)
        center = centers[idx]
        img -= center[2]
        img = np.maximum(img, -150)
        img = np.minimum(img, 150)
        img /= 150
        img += 1
        img *= 255
        img /= 2

        cords_d = np.reshape(cords_d, (-1,3))
        cords_3d = util.pixel2world(cords_d, 'fpad')
        cords_c, skel_camcoords = util.world2pixel(cords_3d, 'fpac')
        cords_3d = np.reshape(cords_3d, (480,640,-1))

        img_rgbd = np.zeros((img.shape[0],img.shape[1],4))
        cords_c = np.reshape(cords_c, (480,640,3))
        img_rgbd = image_loops.color_map(img, cimg, img_rgbd, cords_c)
        img_rgbd = np.asarray(img_rgbd)
        img = np.asarray(img_rgbd[:,:,:3]).copy()

        pred, skel_camcoords = util.world2pixel(preds[idx], 'fpad')
        label, skel_camcoords = util.world2pixel(lbls[idx], 'fpad')

        points = centers[idx]
        img = util.draw_pose('fpad', img, pred, 3, (0,0,255))
        img = cv2.resize(img, (320,240))

        cv2.imwrite(root_dir+'samples/rgbd/predictions/'+str(idx)+'.png', img)

        video_name = root_dir+'samples/rgbd/predictions/videos/'+action+'.avi'


image_folder = root_dir+'samples/rgbd/predictions/'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = [img.replace('.png','') for img in images]
images.sort(key=float)
images = [img+'.png' for img in images]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 25, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    os.remove(os.path.join(image_folder, image))

cv2.destroyAllWindows()
video.release()

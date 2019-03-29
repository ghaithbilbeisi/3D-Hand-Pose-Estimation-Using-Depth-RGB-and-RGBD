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
## This is the preprocessing and validation script for Depth.
## Keep the segment you would like to use and comment out the rest
## this is because there are multiple data files (labels before/after normalization) that will cause conflicts
########################

dataset = 'fpad'
phase = 'train'   ## test/train
root_dir = '/home/bilbeisi/REN/'


############################ Draw pose on depth samples #################################
### draw pose on some depth samples to validate world2pixel and image/label loading
### this segment is only for validation
##############################################################################################
lbls = util.load_labels(dataset,phase) ### load test/train data
names = util.load_names(dataset,phase)
centers = util.load_centers(dataset,phase).astype(float)


for idx, name in enumerate(names):
    if idx%1000 == 0:
        lbl = np.asarray(np.reshape(lbls[idx].split(), (21,3)), dtype=np.float32)
        lbl, skel_camcoords = util.world2pixel(lbl, dataset)
        img = util.load_image(dataset, os.path.join(root_dir, name))
        img /= 1160
        img *= 255
        points = centers[idx]
        img = util.draw_pose(dataset, img, lbl, 3, (255,0,0), points)
        cv2.imwrite(root_dir+'samples/depth/'+phase+'_'+str(idx)+'.png', img)
##############################################################################################



############################# Normalize Labels #################################
### Make sure centers are calculated using get_centers.py before running this
### Also make sure you are working with original non normalized 3D joint coordinates not the normalized ones.
### After running this segment you should rename the original and give this new file the original's name so that it becomes the labels file used moving forward
################################################################################
lbls = util.load_labels(dataset,phase) # load original test/train data
centers = util.load_centers(dataset,phase).astype(float)
fx, fy, ux, uy =  util.get_param(dataset)
out_file = root_dir+'labels/fpad_'+phase+'_label_nm.txt'

lbls = np.asarray([s.split() for s in lbls], dtype=np.float32)

for lid, lbl in enumerate(lbls):
    joints = np.asarray(np.reshape(lbl, (21,3)), dtype=np.float32)
    joints, skel_camcoords = util.world2pixel(joints, dataset)
    lbls[lid] = np.reshape(joints, (63))

lbls = np.reshape(lbls,(-1,63))

x = util.normalize_pose(dataset, lbls, centers, 150, fx, fy)

util.save_results(x, out_file)
################################################################################



############################ Test Normalized joints: norm to 2D pixel to 3D World back to 2D pixel and plot #################################
### Test label normalization by projecting the normalized joints onto some depth image samples
### this segment is only for validation
#############################################################################################################################################
lbls = util.load_labels(dataset,phase) ### load test/train data
names = util.load_names(dataset,phase)
centers = util.load_centers(dataset,phase).astype(float)
fx, fy, ux, uy =  util.get_param(dataset)

lbls = [s.split() for s in lbls]
lbls = np.reshape(np.asarray(lbls, dtype=np.float32),(-1,63))
lbls = util.transform_pose(dataset, lbls, centers, 150, fx, fy) # norm to 2D pixel


for idx, name in enumerate(names):
    if idx%1000 == 0:
        lbl = util.pixel2world(lbls[idx], dataset) # pixel to 3D world
        lbl, skel_camcoords = util.world2pixel(lbl, dataset) # back to 2d pixel from 3D world
        img = util.load_image(dataset, os.path.join(root_dir, name))
        img /= np.max(img)
        img *= 255
        points = centers[idx]
        img = util.draw_pose(dataset, img, lbl, 3, (0,255,0), points)
        cv2.imwrite(root_dir+'samples/depth/from_norm/'+phase+'_'+str(idx)+'.png', img)
        plt = util.plot_joints(dataset, os.path.join(root_dir, name), lbl, skel_camcoords, points)
        plt.savefig(root_dir+'samples/depth/from_norm/'+phase+'_'+str(idx)+'_plot.png')
#############################################################################################################################################



########################### Resize Depth images to match input size #################################
### Crop the depth images using the crop size in mm and store in cropped directory
#####################################################################################################
cropped = 'cropped/'    ### directory for the cropped images

names = util.load_names(dataset,phase)
centers = util.load_centers(dataset,phase).astype(float)
cube_size = 150     # crop size in mm

for idx, name in enumerate(names):
    img = util.load_image(dataset, os.path.join(root_dir, name))
    center = centers[idx]

    crop = crop_image(img, center, dataset)

    crop -= center[2]
    crop = np.maximum(crop, -cube_size)
    crop = np.minimum(crop, cube_size)
    crop /= cube_size
    ### depth values are stored temporarily in [0,255] in order to enable viewing/validating
    crop += 1
    crop *= 255
    crop /= 2

    cv2.imwrite(os.path.join(root_dir,cropped,name), crop)

    if idx % 500 == 0:
        print('{}/{}'.format(idx + 1, len(names)))
#####################################################################################################



############################ Draw pose on cropped depth samples from normalized labels #################################
### Plot the normalized labels on a sample of depth images to validate cropping and label normalization
### this segment is only for validation
########################################################################################################################
lbls = util.load_labels(dataset,phase) ### load test/train data
names = util.load_names(dataset,phase)
centers = util.load_centers(dataset,phase).astype(float)
fx, fy, ux, uy =  util.get_param(dataset)

lbls = [s.split() for s in lbls]
lbls = np.reshape(np.asarray(lbls, dtype=np.float32),(-1,21,3))
lbls[:, :, :2] = (lbls[:, :, :2] * 96)/2 + (96/2)

for idx, name in enumerate(names):
    if idx%1000 == 0:
        img = cv2.imread(os.path.join(root_dir+'cropped', name), 1)
        img = img.astype(float)
        img = util.draw_pose(dataset, img, lbls[idx], 2, (0,255,0))
        img = cv2.resize(img, (200,200))
        cv2.imwrite(root_dir+'samples/depth/cropped/'+phase+'_'+str(idx)+'.png', img)
########################################################################################################################



############################ Draw predicted pose on depth samples and create sample videos of predictions #################################
### Create sample Depth videos from the predicted poses
### this segment is only for validation
###########################################################################################################################################
phase = 'test'
lbls = util.load_labels('fpad', phase) ### load test/train data
names = util.load_names('fpad', phase)
centers = util.load_centers('fpad', phase).astype(float)
lbls, preds = util.load_logs('fpad', 'fpad_test_b159_lr_1e-2_xyz_.txt', centers) #### name of test logfile

for idx, name in enumerate(names):
    action = name.split('/')[2]
    subject = name.split('/')[1]
    if action == 'close_juice_bottle' and subject == 'Subject_2':                  # use this condition to select a certain action by name
    # if idx in (11000, 13000, 15000, 16000, 18000, 21000, 22000, 23000, 31000):   # or this condition to select specific frames or sequences by ID
        pred, skel_camcoords = util.world2pixel(preds[idx], 'fpad')
        label, skel_camcoords = util.world2pixel(lbls[idx], 'fpad')
        img = util.load_image('fpad', os.path.join(root_dir, name))
        img = cv2.imread(os.path.join(root_dir, name), 2)
        img = img.astype(float)
        max = np.max(img)
        img /= 1160
        img *= 255
        im = np.zeros((480,640,3))
        im[:,:,0] = img
        im[:,:,1] = img
        im[:,:,2] = img
        points = centers[idx]
        im = util.draw_pose('fpad', im, pred, 3, (0,0,255))
        im = cv2.resize(im, (320,240))
        cv2.imwrite(root_dir+'samples/depth/predictions/'+str(idx)+'.png', im)
        video_name = root_dir+'samples/depth/predictions/videos/'+action+'.avi'


image_folder = root_dir+'samples/depth/predictions/'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = [img.replace('.png','') for img in images]
images.sort(key=float)
images = [img+'.png' for img in images]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 25, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))############ comment this line to create sample images only
    os.remove(os.path.join(image_folder, image))              ############ comment this line out to also keep the images

cv2.destroyAllWindows()
video.release()
###########################################################################################################################################

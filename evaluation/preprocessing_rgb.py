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
## This is the preprocessing and validation script for RGB.
## Keep the segment you would like to use and comment out the rest
## this is because there are multiple data files (labels before/after normalization) that will cause conflicts
## Some directories may need to be created before running some validation segments if they do not exist
########################

dataset = 'fpac'
phase = 'train'
root_dir = '/home/bilbeisi/REN/'


# ############################ Draw pose on RGB samples #################################
# ### draw pose on some RGB samples to validate world2pixel and image/label loading
# ### this segment is only for validation
# #######################################################################################
# lbls = util.load_labels(dataset,phase) ### load test/train data
# names = util.load_names(dataset,phase)
# centers = util.load_centers(dataset,phase).astype(float)
# centers = np.reshape(centers, (-1,3))
#
# for idx, name in enumerate(names):
#     if idx%1000 == 0:
#         lbl = np.asarray(np.reshape(lbls[idx].split(), (21,3)), dtype=np.float32)
#         lbl, skel_camcoords = util.world2pixel(lbl, dataset)
#         img = util.load_image(dataset, os.path.join(root_dir, name))
#         points = centers[idx]
#         img = util.draw_pose(dataset, img, lbl, 6, (0,255,255), points)
#         cv2.imwrite(root_dir+'samples/rgb/'+phase+'_'+str(idx)+'.png', img)
#         plt = util.plot_joints(dataset, os.path.join(root_dir, name), lbl, skel_camcoords, points)
#         plt.savefig(root_dir+'samples/rgb/'+phase+'_'+str(idx)+'_plot.png')
# #######################################################################################



# ############################# Normalize Labels #################################
# ### Make sure centers are calculated using get_centers.py before running this
# ### Also make sure you are working with original non normalized 3D joint coordinates not the normalized ones.
# ### After running this segment I rename the original and this new file so that it becomes the labels file used moving forward
# ################################################################################
# lbls = util.load_labels(dataset,phase) ### load test/train data
# centers = util.load_centers(dataset,phase).astype(float)
# fx, fy, ux, uy =  util.get_param(dataset)
# out_file = root_dir+'labels/fpac_'+phase+'_label_nm.txt'
#
# lbls = np.asarray([s.split() for s in lbls], dtype=np.float32)
#
# for lid, lbl in enumerate(lbls):
#     joints = np.asarray(np.reshape(lbl, (21,3)), dtype=np.float32)
#     joints, skel_camcoords = util.world2pixel(joints, dataset)
#     lbls[lid] = np.reshape(joints, (63))
#
# lbls = np.reshape(lbls,(-1,63))
#
# x = util.normalize_pose(dataset, lbls, centers, 150, fx, fy)
#
# util.save_results(x, out_file)
# ################################################################################



########################### Test RGB Normalized joints: norm to 2D pixel to 3D World back to 2D pixel and plot #################################
### Test label normalization by projecting the normalized joints onto some RGB image samples
### this segment is only for validation
#############################################################################################################################################
lbls = util.load_labels(dataset,phase) ### load test/train data
names = util.load_names(dataset,phase)
centers = util.load_centers(dataset,phase).astype(float)
fx, fy, ux, uy =  util.get_param(dataset)

lbls = [s.split() for s in lbls]
lbls = np.reshape(np.asarray(lbls, dtype=np.float32),(-1,63))
lbls = util.transform_pose(dataset, lbls, centers, 150, fx, fy) # norm to 2D pixel

centers = np.reshape(centers,(-1,3))

for idx, name in enumerate(names):
    if idx%1000 == 0:
        lbl = util.pixel2world(lbls[idx], dataset) # pixel to 3D world
        lbl, skel_camcoords = util.world2pixel(lbl, dataset) # back to 2d pixel from 3D world
        img = util.load_image(dataset, os.path.join(root_dir, name))
        points = centers[idx]
        img = util.draw_pose(dataset, img, lbl, 5, (0,255,0), points)
        cv2.imwrite(root_dir+'samples/rgb/from_norm/'+phase+'_'+str(idx)+'.png', img)
        plt = util.plot_joints(dataset, os.path.join(root_dir, name), lbl, skel_camcoords, points)
        plt.savefig(root_dir+'samples/rgb/from_norm/'+phase+'_'+str(idx)+'_plot.png')
#############################################################################################################################################




############################ Resize RGB images to match input size #################################
### Crop the RGB images using the crop size in mm and store in cropped directory
#####################################################################################################
cropped = 'cropped/'
names = util.load_names(dataset,phase)
centers = util.load_centers(dataset,phase).astype(float)
centers = np.reshape(centers, (-1,3))

for idx, name in enumerate(names):
    img = util.load_image(dataset, os.path.join(root_dir, name))
    crop = crop_image(img, centers[idx], dataset)

    name = name.replace('.jpeg','.png')
    cv2.imwrite(os.path.join(root_dir,cropped,name), crop)
    if idx % 500 == 0:
        print('{}/{}'.format(idx + 1, len(names)))
#####################################################################################################



############################ Draw pose on cropped RGB samples from normalized labels #################################
### Plot the normalized labels on a sample of RGB images to validate cropping and label normalization
### this segment is only for validation
######################################################################################################################
lbls = util.load_labels(dataset,phase) ### load test/train data
names = util.load_names(dataset,phase)
fx, fy, ux, uy =  util.get_param(dataset)

lbls = [s.split() for s in lbls]
lbls = np.reshape(np.asarray(lbls, dtype=np.float32),(-1,21,3))
lbls[:, :, :2] = (lbls[:, :, :2] * 96)/2 + (96/2)

for idx, name in enumerate(names):
    if idx%1000 == 0:
        name = name.replace('.jpeg','.png')
        img = cv2.imread(os.path.join(root_dir+'cropped', name), 1)
        img = img.astype(float)
        img = util.draw_pose(dataset, img, lbls[idx], 2, (255,0,0))
        img = cv2.resize(img, (200,200))
        cv2.imwrite(root_dir+'samples/rgb/cropped/'+phase+'_'+str(idx)+'.png', img)
########################################################################################################################



############################ Draw predicted pose on RGB samples ###########################################################################
### Create sample RGB videos from the predicted poses
### this segment is only for validation
###########################################################################################################################################
phase = 'test'
lbls = util.load_labels('fpac', phase) ### load test/train data
names = util.load_names('fpac', phase)
centers = util.load_centers('fpac', phase).astype(float)
lbls, preds = util.load_logs('fpac', 'fpac_test_b53_lr_1e-2_xyz_20k_.txt', centers)

for idx, name in enumerate(names):
    action = name.split('/')[2]
    subject = name.split('/')[1]
    if action == 'close_juice_bottle' and subject == 'Subject_2':                  # use this condition to select a certain action by name
    # if idx in (11000, 13000, 15000, 16000, 18000, 21000, 22000, 23000, 31000):   # or this condition to select specific frames or sequences by ID
        pred, skel_camcoords = util.world2pixel(preds[idx], 'fpac')
        label, skel_camcoords = util.world2pixel(lbls[idx], 'fpac')
        img = util.load_image('fpac', os.path.join(root_dir, name))
        img = cv2.imread(os.path.join('/home/bilbeisi/REN', name), 1)
        img = img.astype(float)

        points = centers[idx]
        img = util.draw_pose('fpac', img, pred, 6, (0,0,255))
        img = cv2.resize(img, (480,270))
        cv2.imwrite(root_dir+'samples/rgb/predictions/'+str(idx)+'.png', img)

        video_name = root_dir+'samples/rgb/predictions/videos/'+action+'.avi'

image_folder = root_dir+'samples/rgb/predictions/'

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

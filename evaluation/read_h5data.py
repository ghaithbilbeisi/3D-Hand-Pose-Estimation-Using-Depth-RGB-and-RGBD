import numpy as np
import h5py
import cv2
import scipy.io as sio
import sys
import os
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy import ndimage
from scipy.misc import toimage
import copy
import glob
from numpy import loadtxt

import util

def show_3d(ax, points, c = (255, 0, 0)):
  points = points.reshape(J, 3)
  x, y, z = np.zeros((3, J))
  for j in range(J):
    x[j] = points[j, 0]
    y[j] = - points[j, 1]
    z[j] = - points[j, 2]
  ax.scatter(z, x, y, c = c)

def readHData(fnames):
    image_data=None
    joints_data = None
    i = 0
    name=fnames
    f = h5py.File(name, 'r')
    image_key = list(f.keys())[0]
    joints_key = list(f.keys())[1]
    if i ==0:
        image_data = np.array(f[image_key])
        joints_data = np.array(f[joints_key])
    else:
        image_data = np.concatenate((image_data,np.array(f[image_key])),axis=0)
        joints_data = np.concatenate((joints_data,np.array(f[joints_key])),axis=0)
    i = i+1

    print (image_data.shape)
    print (joints_data.shape)

    return image_data, joints_data

################################################ RGBD  #######################################33
root_dir = '/home/bilbeisi/REN/'
dataset = 'rgbd'    # confirm before running the script! one of fpad/fpac/rgbd
J = 21
# TRAIN SET
# In the case of RGBD only run one of trian/test at a time to avoid a memory error
train_d, train_j = readHData('../labels/'+dataset+'_train_data.h5')
# TEST SET
test_d, test_j = readHData('../labels/'+dataset+'_test_data.h5')

for i in range(train_d.shape[0]):
    if i % 1000 == 0:
        x1 = train_d[i,:,:,:]
        if dataset in ('rgbd','fpac'):
            x1 = np.swapaxes(x1, 2, 0)
        x1 += 1
        x1 *= 255
        x1 /= 2
        y1 = train_j[i,:]
        y1 = y1.reshape(J,3)
        y1 = (y1+1)/2 * 96
        if dataset == 'fpad':
            x1 = x1.reshape(96,96,1)
            img = cv2.cvtColor(x1, cv2.COLOR_GRAY2RGB)
        else:
            img = x1[:,:,:3].copy()
        if dataset == 'rgbd':
            img = util.draw_pose('fpad', img, y1, 2, (0,0,255))
        else:
            img = util.draw_pose(dataset, img, y1, 2, (0,0,255))
        img = cv2.resize(img, (200,200))
        cv2.imwrite(root_dir+'trial/'+dataset+'/train_'+str(i)+'.png', img)

for i in range(test_d.shape[0]):
    if i % 1000 == 0:
        x2 = test_d[i,:,:,:]
        if dataset in ('rgbd','fpac'):
            x2 = np.swapaxes(x2, 2, 0)
        x2 += 1
        x2 *= 255
        x2 /= 2
        y2 = test_j[i,:]
        y2 = y2.reshape(J,3)
        y2 = (y2+1)/2 * 96
        if dataset == 'fpad':
            x2 = x2.reshape(96,96,1)
            img = cv2.cvtColor(x2, cv2.COLOR_GRAY2RGB)
        else:
            img = x2[:,:,:3].copy()
        if dataset == 'rgbd':
            img = util.draw_pose('fpad', img, y2, 2, (0,0,255))
        else:
            img = util.draw_pose(dataset, img, y2, 2, (0,0,255))
        img = cv2.resize(img, (200,200))
        cv2.imwrite(root_dir+'trial/'+dataset+'/test_'+str(i)+'.png', img)

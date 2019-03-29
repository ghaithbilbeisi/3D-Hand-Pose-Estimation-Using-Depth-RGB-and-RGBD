import cv2
import numpy as np
import cython

@cython.boundscheck(False)
cpdef double[:, :, :] depth2cords(double[:, :] d_img, double[:, :, :] img_cords):
    # set the variable extension types
    cdef int x, y, w, h

    # grab the image dimensions
    h = d_img.shape[0]
    w = d_img.shape[1]
    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            img_cords[y,x][0] = x
            img_cords[y,x][1] = y
            img_cords[y,x][2] = d_img[y,x]
    return img_cords

@cython.boundscheck(False)
cpdef double[:, :, :] color_map(double[:, :] d_img, double[:, :, :] c_img, double[:, :, :] rgbd_img, double[:, :, :] c_cords):
    # set the variable extension types
    cdef int x, y, w, h

    # grab the image dimensions
    h = rgbd_img.shape[0]
    w = rgbd_img.shape[1]
    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            c0 = int(c_cords[y,x][0])
            c1 = int(c_cords[y,x][1])
            if c1>=1080 or c0>=1920 or c1<0 or c0<0:
                rgbd_img[y,x][0] = d_img[y,x]
                rgbd_img[y,x][1] = d_img[y,x]
                rgbd_img[y,x][2] = d_img[y,x]
                rgbd_img[y,x][3] = d_img[y,x]
            else:
                rgbd_img[y,x][0] = c_img[c1,c0][0]
                rgbd_img[y,x][1] = c_img[c1,c0][1]
                rgbd_img[y,x][2] = c_img[c1,c0][2]
                rgbd_img[y,x][3] = d_img[y,x]
    return rgbd_img

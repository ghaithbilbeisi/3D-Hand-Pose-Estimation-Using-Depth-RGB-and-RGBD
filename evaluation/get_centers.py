# done, not tested
import os
import sys
import cv2
import util
import numpy as np


def print_usage():
    print('usage: {} fpad/fpac base_dir out_file test/train'.format(sys.argv[0]))
    exit(-1)


def save_results(results, out_file):
    with open(out_file, 'w') as f:
        for result in results:
            for j in range(result.shape[0]):
                for k in range(result.shape[1]):
                    f.write('{:.3f} '.format(result[j, k]))
            f.write('\n')

def main():
    if len(sys.argv) < 4:
        print_usage()

    dataset = sys.argv[1]
    base_dir = sys.argv[2]
    out_file = sys.argv[3]
    phase = sys.argv[4]
    names = util.load_names(dataset, phase)
    lbls = util.load_labels(dataset, phase)
    centers = []


    for idx, strin in enumerate(lbls):
        # load label data
        joints = np.asarray(np.reshape(strin.split(), (21,3)), dtype=np.float32)
        # convert label data from world coordinates to pixel locations
        joints, skel_camcoords = util.world2pixel(joints, dataset)
        # calculate centers
        c = util.get_center_fpad(joints)
        c = np.asarray(c, dtype=np.float32)
        centers.append(c.reshape((1,3)))
        if idx % 500 == 0:
            print('{}/{}'.format(idx + 1, len(names)))


        centers.append(center.reshape((1, 3)))
        if idx % 500 == 0:
            print('{}/{}'.format(idx + 1, len(names)))
    util.save_results(centers, out_file)

if __name__ == '__main__':
    main()

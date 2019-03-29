import cv2
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
from PIL import Image


def get_positions(in_file):
    with open(in_file) as f:
        positions = [list(map(float, line.strip().split())) for line in f]
    return np.reshape(np.array(positions), (-1, len(positions[0]) / 3, 3))


def check_dataset(dataset):
    return dataset in set([ 'fpad', 'fpac','rgbd'])


def get_param(dataset):
    if dataset == 'fpad':         ######################### Depth ##################################
        return 475.065948, 475.065857, 315.944855, 245.287079
    elif dataset == 'fpac':         ######################### RGB ####################################
        return 1395.749023, 1395.749268, 935.732544, 540.681030


def get_errors(dataset, log_name, phase):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    centers = load_centers(dataset,phase).astype(float)
    labels, outputs = load_logs(dataset, log_name, centers)
    labels = np.reshape(labels, (-1, 21, 3))
    outputs = np.reshape(outputs, (-1, 21, 3))
    params = get_param(dataset)
    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))
    return errors


def load_image(dataset, name, input_size=None, is_flip=False):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    if dataset == 'fpad': ##################### Depth ############################
        img = cv2.imread(name, 2)
        img = img.astype(float)
    elif dataset == 'fpac': ##################### RGB ############################
        img = cv2.imread(name, 1)

    if input_size is not None:
        img = cv2.resize(img, (640, 480))
    if is_flip:
        img[:, ::-1] = img
    return img


def load_names(dataset, phase):    ## read data file names from txt
    with open('../labels/'+dataset+'_'+phase+'_list.txt') as f:
        return [line.strip() for line in f]


def load_labels(dataset, phase):    ## read data file names from txt
    with open('../labels/'+dataset+'_'+phase+'_label.txt'.format(dataset)) as f:
        return [line.strip() for line in f]


def load_centers(dataset, phase):  ## read centers from txt
    with open('../labels/'+dataset+'_'+phase+'_center.txt'.format(dataset)) as f:
        return np.array([line.strip().split() for line in f])


def get_sketch_setting(dataset):
    return [(0, 1), (1, 6), (6, 7), (7, 8), (0, 2), (2, 9), (9, 10), (10, 11),
                (0, 3), (3, 12), (12, 13), (13, 14), (0, 4), (4, 15), (15, 16), (16, 17),
                (0, 5), (5, 18), (18, 19), (19, 20)]


def draw_pose(dataset, img, pose, size=3, color=(0,0,255), points=None):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), size, color, -1)
    for x, y in get_sketch_setting(dataset):
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), color, int(size/2))
    if points is not None:
        # for pt in points:
        cv2.circle(img, (int(points[0]), int(points[1])), 5, color, 3)
    return img


def get_center_fpad(joints):
    #### for new centers (mean dist. btwn wrist and finger tips)####
    # mids = np.zeros((5,3))
    # #for x in range(1,6):
    # c = 0
    # for x in (8,11,14,17,20):
    #     mids[c] = (joints[0] + joints[x]) / 2
    #     c+=1
    mids = np.zeros((5,3))
    for x in range(1,6):
        mids[x-1] = (joints[0] + joints[x]) / 2
    center = [0,0,0]
    center[0:3] = np.mean(mids[0:3], 0)
    return center


def save_results(results, out_file):
    with open(out_file, 'w') as f:
        for result in results:
            for j in range(result.shape[0]):
                for k in range(result.shape[1]):
                    f.write('{} '.format(result[j, k]))
            f.write('\n')


def world2pixel(x, dataset):
    skel = x.reshape(-1, 3)
    if dataset == 'fpac':
        cam_extr = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594,
              25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807,
              3.902], [0, 0, 0, 1]])
        cam_intr = np.array([[1395.749023, 0, 935.732544],
                             [0, 1395.749268, 540.681030], [0, 0, 1]])
    elif dataset == 'fpad':
        cam_extr = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594,
              25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807,
              3.902], [0, 0, 0, 1]])
        cam_intr = np.array([[475.065948, 0, 315.944855],
                             [0, 475.065857, 245.287079], [0, 0, 1]])
    # Apply camera extrinsic to hand skeleton
    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
    skel_camcoords = cam_extr.dot(
        skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

    if dataset == 'fpad':
        skel_hom2d = np.array(cam_intr).dot(skel.transpose()).transpose()
    else:
        skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()

    skel_proj = skel_hom2d
    skel_proj[:, :2] = (skel_hom2d[:, :2] / skel_hom2d[:, 2:])

    return skel_proj, skel_camcoords


def pixel2world(skel_proj, dataset):
    if dataset == 'fpac':
        cam_extr = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594,
              25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807,
              3.902], [0, 0, 0, 1]])
        cam_intr = np.array([[1395.749023, 0, 935.732544],
                             [0, 1395.749268, 540.681030], [0, 0, 1]])
    elif dataset == 'fpad':
        cam_extr = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594,
              25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807,
              3.902], [0, 0, 0, 1]])
        cam_intr = np.array([[475.065948, 0, 315.944855],
                             [0, 475.065857, 245.287079], [0, 0, 1]])

    shape = skel_proj.shape

    skel_hom2d = copy.deepcopy(skel_proj)
    skel_hom2d[:, :2] = (skel_proj[:, :2] * skel_proj[:, 2:])

    if dataset == 'fpad':
        skel = np.linalg.inv(np.array(cam_intr)).dot(skel_hom2d.transpose()).transpose()

    else:
        skel_camcoords = np.linalg.inv(np.array(cam_intr)).dot(skel_hom2d.transpose()).transpose()
        skel_camcoords = np.concatenate([skel_camcoords, np.ones([shape[0], 1])], 1)
        skel_hom = np.linalg.inv(cam_extr).dot(skel_camcoords.transpose()).transpose()
        skel = skel_hom[:, :3].astype(np.float32)

    return skel


def load_logs(dataset, log_name, centers):
    fx, fy, ux, uy = get_param(dataset)
    lblogname = log_name.replace('_.txt','_label.txt')
    predlogname = log_name.replace('_.txt','_predict.txt')

    flbl = open('../logs/{}'.format(lblogname))
    lbls = [float(line.strip().split()[-1]) for line in flbl]
    lbls = np.reshape(np.asarray(lbls, dtype=np.float32),(-1,63))
    lbls = transform_pose(dataset, lbls, centers, 150, fx, fy)
    for lidx, lbl in enumerate(lbls):
        lbls[lidx] = pixel2world(lbl, dataset)

    fpred = open('../logs/{}'.format(predlogname))
    preds = [float(line.strip().split()[-1]) for line in fpred]
    preds = np.reshape(np.asarray(preds, dtype=np.float32),(-1,63))
    preds = transform_pose(dataset, preds, centers, 150, fx, fy)
    for pidx, pred in enumerate(preds):
        preds[pidx] = pixel2world(pred, dataset)

    return lbls, preds


# Display utilities
def plot_joints(dataset, name, lbl, skel_camcoords, points=None):
    fig = plt.figure()
    ax = fig.add_subplot(221)
    img = Image.open(name)

    if dataset == 'fpad':
        img = np.array(img)
        img = img.astype(float)
        img /= 1160
        img *= 255
        img = Image.fromarray(img)

    ax.imshow(img)
    visualize_joints_2d(ax, lbl, joint_idxs=False)
    if points is not None:
        ax.scatter(points[0],points[1],10,'k')
        # cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (255, 255, 0), -1)
    for proj_idx, (proj_1, proj_2) in enumerate([[0, 1], [1, 2], [0, 2]]):
        ax = fig.add_subplot(2, 2, 2 + proj_idx)
        if proj_idx == 0:
            # Invert y axes to align with image in camera projection
            ax.invert_yaxis()
        ax.set_aspect('equal')
        visualize_joints_2d(
        ax,
        np.stack(
            [skel_camcoords[:, proj_1], skel_camcoords[:, proj_2]],
            axis=1),
        joint_idxs=False)
    return plt


def visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 6, 7, 8), (0, 2, 9, 10, 11), (0, 3, 12, 13, 14),
                 (0, 4, 15, 16, 17), (0, 5, 18, 19, 20)]

    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha)



def _draw2djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)



def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
alpha=alpha)


def normalize_pose(dataset, poses, centers, cube_size, fx, fy):
    cube_size = 150
    res_poses = np.asarray(poses, dtype=np.float32)
    num_joint = int(poses.shape[1] / 3)
    centers_tile = np.tile(centers, (num_joint, 1, 1)).transpose([1, 0, 2])

    res_poses[:, 2::3] -= centers_tile[:, :, 2]
    res_poses[:, 1::3] = (res_poses[:, 1::3] - centers_tile[:, :, 1]) * centers_tile[:, :, 2] / fy
    res_poses[:, 0::3] = (res_poses[:, 0::3] - centers_tile[:, :, 0]) * centers_tile[:, :, 2] / fx

    res_poses = np.reshape(res_poses, [poses.shape[0], -1, 3])
    res_poses = res_poses / cube_size

    return res_poses


def transform_pose(dataset, poses, centers, cube_size, fx, fy):
    res_poses = np.array(poses) * cube_size
    num_joint = int(poses.shape[1] / 3)
    centers_tile = np.tile(centers, (num_joint, 1, 1)).transpose([1, 0, 2])

    res_poses[:, 0::3] = res_poses[:, 0::3] * fx / centers_tile[:, :, 2] + centers_tile[:, :, 0]
    res_poses[:, 1::3] = res_poses[:, 1::3] * fy / centers_tile[:, :, 2] + centers_tile[:, :, 1]
    res_poses[:, 2::3] += centers_tile[:, :, 2]
    res_poses = np.reshape(res_poses, [poses.shape[0], -1, 3])

    return res_poses


def crop_image(img, center, dataset, is_debug=False):
    fx, fy, ux, uy = get_param(dataset)
    cube_size = 150
    input_size = 96

    xstart = center[0] - cube_size / center[2] * fx # _fx/_fy util.get_params
    xend = center[0] + cube_size / center[2] * fx # cube_size = 150
    ystart = center[1] - cube_size / center[2] * fy
    yend = center[1] + cube_size / center[2] * fy

    src = [(xstart, ystart), (xstart, yend), (xend, ystart)]
    dst = [(0, 0), (0, input_size - 1), (input_size - 1, 0)] # input_size = 96
    trans = cv2.getAffineTransform(np.array(src, dtype=np.float32),
            np.array(dst, dtype=np.float32))
    res_img = cv2.warpAffine(img, trans, (input_size, input_size), None,
            cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0).astype(np.float32)

    return res_img

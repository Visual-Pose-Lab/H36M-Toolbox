import numpy as np
import os.path as osp
from scipy.io import loadmat
from subprocess import call
from os import makedirs
import pickle
from img import crop_image
import cv2
import os
from tqdm import tqdm

camera_index = {
    0:54138969,
    1:55011271,
    2:58860488,
    3:60457274
}

image_shape = (192, 256)

from metadata import load_h36m_metadata
metadata = load_h36m_metadata()


def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[0] -= 1000.0
    tl_joint[1] -= 900.0
    br_joint = root_joint.copy()
    br_joint[0] += 1000.0
    br_joint[1] += 1100.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d

makedirs('/home/ltf/dataset/croped', exist_ok=True)

def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

# cnt = 0

# print('Loading 2D detections...')
pose2dgt = np.load('/home/ltf/data/xw/MHFormer/dataset/h36m/data_2d_h36m_gt.npz', allow_pickle=True)
pose2dgt_metadata = pose2dgt['metadata'].item()
pose2dgt = pose2dgt['positions_2d'].item()


pose2dcpn = np.load('/home/ltf/data/xw/MHFormer/dataset/h36m/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)
pose2dcpn = pose2dcpn['positions_2d'].item()

image_paths = np.load('crop_image_paths.npz', allow_pickle=True)
image_paths = image_paths['image_path'].item()

pose2dgt_norm = {}
pose2dcpn_norm = {}

for sub in pose2dgt.keys():
    pose2dgt_norm[sub] = {}
    pose2dcpn_norm[sub] = {}
    for act in pose2dgt[sub].keys():
        pose2dgt_norm_ = []
        pose2dcpn_norm_ = []
        for ca in range(len(pose2dgt[sub][act])):
            nposes = len(pose2dgt[sub][act][ca])
            print('crop image on subject: {} action: {} view: {}'.format(sub, act, ca))

            image_path = image_paths[sub][act][ca][0]
            image = cv2.imread(os.path.join('/home/ltf/dataset/', image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            h, w, _ = image.shape
            
            pose2dgt_norm_.append(normalize_screen_coordinates(pose2dgt[sub][act][ca][..., :2], w=w, h=h))
            pose2dcpn_norm_.append(normalize_screen_coordinates(pose2dcpn[sub][act][ca][..., :2], w=w, h=h))
        pose2dgt_norm[sub][act] = pose2dgt_norm_
        pose2dcpn_norm[sub][act] = pose2dcpn_norm_


print('Saving...')
metadata = {
    'num_joints': pose2dgt_metadata['num_joints'],
    'keypoints_symmetry': pose2dgt_metadata['keypoints_symmetry']
}
np.savez_compressed('/home/ltf/dataset/data_2d_h36m_gt_norm.npz', positions_2d=pose2dgt_norm, metadata=metadata)
np.savez_compressed('/home/ltf/dataset/data_2d_h36m_cpn_norm.npz', positions_2d=pose2dcpn_norm, metadata=metadata)

print('Done.')




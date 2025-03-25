import pickle
import h5py
import cv2
import numpy as np
import os.path as osp
from scipy.io import loadmat
from subprocess import call
import os
from os import makedirs
# from spacepy import pycdf
from tqdm import tqdm
import cdflib
import sys
from common.camera import *

from transform import get_affine_transform, affine_transform, \
    normalize_screen_coordinates, _infer_box, _weak_project 
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

if __name__ == '__main__':
    subject_list = [1, 5, 6, 7, 8, 9, 11]
    action_list = [x for x in range(2, 17)]
    subaction_list = [x for x in range(1, 3)]
    camera_list = [x for x in range(1, 5)]

    train_list = [1, 5, 6, 7, 8]
    test_list = [9, 11]

    joint_idx = [0, 1, 2, 3, 6, 7, 8, 12, 16, 14, 15, 17, 18, 19, 25, 26, 27]

    camera_index = {
        0:54138969,
        1:55011271,
        2:58860488,
        3:60457274
    }

    print('Loading 2D detections...')
    keypoints = np.load('/home/ltf/data/xw/MHFormer/dataset/h36m/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    keypoints = keypoints['positions_2d'].item()

    with open('camera_data.pkl', 'rb') as f:
        camera_data = pickle.load(f)

    # data_2d_h36m_gt_crop = {}
    # data_2d_h36m_cpn_crop = {}
    # data_3d_h36m_uvd = {}
    data_image_paths = {}
    for sub in keypoints.keys() - ['S2', 'S3', 'S4']:
        # data_2d_h36m_gt_crop[sub] = {}
        # data_2d_h36m_cpn_crop[sub] = {}
        # data_3d_h36m_uvd[sub] = {}
        data_image_paths[sub] = {}
        for act in  keypoints[sub].keys():
            # pose_2d_h36m_gt_crop = []
            # pose_2d_h36m_cpn_crop = []
            # pose_3d_h36m_uvd = []
            pose_image_paths = []
            for ca in range(len(keypoints[sub][act])):
                
                # camera = camera_data[(int(sub[1:]), ca+1)]
                # camera_dict = {}
                # camera_dict['R'] = camera[0]
                # camera_dict['T'] = camera[1]
                # camera_dict['fx'] = camera[2][0]
                # camera_dict['fy'] = camera[2][1]
                # camera_dict['cx'] = camera[3][0]
                # camera_dict['cy'] = camera[3][1]
                # camera_dict['k'] = camera[4]
                # camera_dict['p'] = camera[5]

                annotname = '{}.{}.cdf'.format(act, camera_index[ca])

                if  sub == 'S1':
                    if "Photo" in act:
                        annotname = annotname.replace("Photo", "TakingPhoto")
                    elif "WalkDog" in act:
                        annotname = annotname.replace("WalkDog", "WalkingDog")

                # annofile3d = osp.join('extracted', subject, 'Poses_D3_Positions_mono_universal', annotname)
                annofile3d_camera = osp.join('/home/ltf/data/DATASETs/H36M/', sub, 'Poses_D3_Positions_mono', annotname)
                annofile2d = osp.join('/home/ltf/data/DATASETs/H36M/', sub, 'Poses_D2_Positions', annotname)

                # with pycdf.CDF(annofile3d) as data:
                # data = cdflib.CDF(annofile3d)
                # pose3d = np.array(data.varget("Pose"))
                # pose3d = np.reshape(pose3d, (-1, 32, 3))

                # with pycdf.CDF(annofile3d_camera) as data:
                data = cdflib.CDF(annofile3d_camera)
                pose3d_camera = np.array(data.varget("Pose"))
                pose3d_camera = np.reshape(pose3d_camera, (-1, 32, 3))#[:, joint_idx] / 1000.0



                # with pycdf.CDF(annofile2d) as data:
                data = cdflib.CDF(annofile2d)
                pose2d_gt = np.array(data.varget("Pose"))
                pose2d_gt = np.reshape(pose2d_gt, (-1, 32, 2))
                pose2d_cpn = keypoints[sub][act][ca]

                nposes = min(pose3d_camera.shape[0], pose2d_gt.shape[0])
                # if pose2d_cpn.shape[0] > nposes:
                #     pose2d_cpn = pose2d_cpn[:nposes]
                # assert pose2d_gt.shape[0] == pose2d_cpn.shape[0] 
                # assert pose2d_cpn.shape[0] == pose3d_camera.shape[0]

                image_format = 'images/{}/{}/{}/{}_{}_{}_{:06d}.jpg'

                # pose2d_gt_crop = np.zeros([nposes, 17, 2], dtype='float32')
                # pose2d_cpn_crop = np.zeros([nposes, 17, 2], dtype='float32')

                image_path_ = []

                print('processing on subject: {} action: {} view: {} len: {}'.format(sub, act, ca, nposes))

                for i in tqdm(range(nposes)):
                    
                #     box = _infer_box(pose3d_camera[i, joint_idx, :], camera_dict, 0)
                #     center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                #     scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)

                #     trans = get_affine_transform(center, scale, 0, [192, 256])

                #     for j in range(17):
                #         pose2d_gt_crop[i][j] = affine_transform(pose2d_gt[i, joint_idx, :][j], trans)
                #         pose2d_cpn_crop[i][j] = affine_transform(pose2d_cpn[i][j], trans)

                    image_path_.append(image_format.format(sub, act, ca, sub, act, ca, i+1))
                    # print((image_format.format(sub, act, ca, sub, act, ca, i+1)))

                # pose_2d_h36m_gt_crop.append(pose2d_gt_crop)
                # pose_2d_h36m_cpn_crop.append(pose2d_cpn_crop)
                # pose_3d_h36m_uvd.append(pose3d_camera[:, joint_idx, :]/1000.0)
                pose_image_paths.append(image_path_)
            
            # data_2d_h36m_gt_crop[sub][act] = pose_2d_h36m_gt_crop
            # data_2d_h36m_cpn_crop[sub][act] = pose_2d_h36m_cpn_crop
            # data_3d_h36m_uvd[sub][act] = pose_3d_h36m_uvd
            data_image_paths[sub][act] = pose_image_paths

    print('Saving...')
    metadata = {
        'num_joints': keypoints_metadata['num_joints'],
        'keypoints_symmetry': keypoints_metadata['keypoints_symmetry']
    }
    # np.savez_compressed('data_2d_h36m_gt_crop.npz', positions_2d=data_2d_h36m_gt_crop, metadata=metadata)
    # np.savez_compressed('data_2d_h36m_cpn_crop.npz', positions_2d=data_2d_h36m_cpn_crop, metadata=metadata)
    # np.savez_compressed('/home/ltf/dataset/data_3d_h36m_uvd_2.npz', positions_3d=data_3d_h36m_uvd, metadata=metadata)
    np.savez_compressed('crop_image_paths.npz', image_path = data_image_paths, metadata=metadata)
    
    print('Done.')
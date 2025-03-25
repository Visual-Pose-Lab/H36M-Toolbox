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


# cnt = 0

# print('Loading 2D detections...')
# keypoints = np.load('data_2d_h36m_gt_crop.npz', allow_pickle=True)
# keypoints = keypoints['positions_2d'].item()

pose3d = np.load('data_3d_h36m_uvd.npz', allow_pickle=True)
pose3d = pose3d['positions_3d'].item()

image_paths = np.load('crop_image_paths.npz', allow_pickle=True)
image_paths = image_paths['iamge_path'].item() ## 注意 拼错了

with open('camera_data.pkl', 'rb') as f:
    camera_data = pickle.load(f)

for sub in image_paths.keys():
    for act in  image_paths[sub].keys():
        for ca in range(len(image_paths[sub][act])):
            nposes = len(image_paths[sub][act][ca])
            print('crop image on subject: {} action: {} view: {}'.format(sub, act, ca))
            for cnt in tqdm(range(nposes)):

                camera = camera_data[(int(sub[1:]), ca+1)]
                camera_dict = {}
                camera_dict['R'] = camera[0]
                camera_dict['T'] = camera[1]
                camera_dict['fx'] = camera[2][0]
                camera_dict['fy'] = camera[2][1]
                camera_dict['cx'] = camera[3][0]
                camera_dict['cy'] = camera[3][1]
                camera_dict['k'] = camera[4]
                camera_dict['p'] = camera[5]

                image_path = image_paths[sub][act][ca][cnt]
                image = cv2.imread(os.path.join('/home/ltf/dataset/', image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

                # crop image

                box = _infer_box(1000 * pose3d[sub][act][ca][cnt], camera_dict, 0)
                center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)

                croped_image = crop_image(image, center, scale, image_shape) # (256, 192, 3) uint8
                # print(image.shape, croped_image.shape)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                save_path = '/home/ltf/dataset/croped/' + image_path
                # print(save_path)
                makedirs(osp.join('/home/ltf/dataset/croped/images', sub, act, str(ca)), exist_ok=True)

                try:
                    cv2.imwrite(save_path, croped_image)
                except Exception as e:
                    print(f"Error saving image to {save_path}: {e}")








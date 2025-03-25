import numpy as np
import os.path as osp
from scipy.io import loadmat
from subprocess import call
from os import makedirs
import pickle

subject_list = [1, 5, 6, 7, 8, 9, 11]
action_list = [x for x in range(2, 17)]
subaction_list = [x for x in range(1, 3)]
camera_list = [x for x in range(1, 5)]
camera_index = {
    0:54138969,
    1:55011271,
    2:58860488,
    3:60457274
}
image_shape = (256, 256, 3)


from metadata import load_h36m_metadata
metadata = load_h36m_metadata()

makedirs('/home/ltf/dataset/images', exist_ok=True)


cnt = 0

print('Loading 2D detections...')
keypoints = np.load('data_2d_h36m_gt_crop.npz', allow_pickle=True)
keypoints = keypoints['positions_2d'].item()

with open('camera_data.pkl', 'rb') as f:
    camera_data = pickle.load(f)

for sub in keypoints.keys():
    for act in  keypoints[sub].keys():
        for ca in range(len(keypoints[sub][act])):
            camera = camera_data[(int(sub[1:]), ca+1)]
            subdir = '{}/{}/{}'.format(sub, act, ca)

            makedirs(osp.join('/home/ltf/dataset/images', subdir), exist_ok=True)

            videoname = '{}.{}.mp4'.format(act, camera_index[ca])
            if  sub == 'S1':
                if "Photo" in act:
                    videoname = videoname.replace("Photo", "TakingPhoto")
                elif "WalkDog" in act:
                    videoname = videoname.replace("WalkDog", "WalkingDog")

            videopath = osp.join('/home/ltf/data/DATASETs/H36M', sub, 'Videos', videoname)

            fileformat = '/home/ltf/dataset/images/' + sub + '/' + act + '/' + str(ca) + '/' + sub + '_' + act + '_' + str(ca) + '_%06d.jpg'

            print(videopath)
            cnt += 1
            call([
                'ffmpeg',
                '-nostats',
                '-i', videopath,
                '-qscale:v', '3',
                fileformat
                    ])
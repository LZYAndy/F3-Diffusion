import cv2
import pandas as pd
import numpy as np
import glob
import json
from tqdm import tqdm
import math
from copy import deepcopy
import os
import warnings
warnings.filterwarnings("ignore")

frames = []
data = json.load(open('./data/finediving/train.json'))
pause = True


COCO_SKELETON = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (5, 6), (11, 12),
    (4, 6), (6, 8), (8, 10), (6, 12), (12, 14), (14, 16),
    (3, 5), (5, 7), (7, 9), (5, 11), (11, 13), (13, 15)
]


# Function to draw skeleton
def draw_skeleton(image, keypoints, skeleton=COCO_SKELETON, joint_radius=5, bone_thickness=2):
    """
    Draw skeletons with joints and bones on an image.

    Args:
        image (np.array): The image on which to draw the skeleton.
        keypoints (np.array): The 2D keypoints, shape (17, 2).
        skeleton (list): List of tuples representing the connections.
        joint_radius (int): Radius of each joint circle.
        bone_thickness (int): Thickness of the lines connecting the joints.
    """

    keypoints = keypoints.astype(np.int32)

    # Draw bones (lines between connected joints)
    for start, end in skeleton:
        if start < len(keypoints) and end < len(keypoints):
            pt1, pt2 = tuple(keypoints[start]), tuple(keypoints[end])
            print(pt1, pt2)
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:  # Valid points
                cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=bone_thickness)

    # Draw joints (keypoints)
    for x, y in keypoints:
        if x > 0 and y > 0:  # Valid keypoint
            cv2.circle(image, (x, y), joint_radius, color=(0, 0, 255), thickness=-1)


for i, clip in enumerate(data):
    print('Clip %d -> %d' % (i, len(data)))
    # start, end = clip['video'].split('_')[-2:]
    # start, end = int(start), int(end)
    file_name = clip['video']
    part1, part2 = file_name.split('__')
    # video_name = '_'.join(clip['video'].split('_')[:-2])
    # video_name = '/mnt/ssd/zhaoyu/F3Set/videos/' + video_name + '.mp4'
    print(file_name)
    # cap = cv2.VideoCapture(video_name)

    if len(clip['events']) == 0:
        continue

    # lines_df = pd.read_csv('tracking_data/%s_line.csv' % file_name)
    # players_df = pd.read_csv('tracking_data/%s_player.csv' % file_name)
    # ball_df = pd.read_csv('tracking_data/%s_ball.csv' % file_name)

    # clip_players = pd.read_csv('players/f3set-tennis/%s.csv' % file_name)
    clip_skeletons = pd.read_pickle('skeletons/finediving/%s.pkl' % file_name)

    # lines = None
    # for idx, t in enumerate(range(start, end)):
    frame_paths = sorted(glob.glob('/home/user/zhaoyu/IJCAI/FINADiving_MTL_256s/%s/%s/*.jpg' % (part1, part2)))
    for idx, frame_path in tqdm(enumerate(frame_paths)):
        frame = cv2.imread(frame_path)  # cap.read()
        # frame = cap.read()
        HT, WD, _ = frame.shape
        label = 'NA'

        # if not clip_players.query('frame==@t').empty:
        #     player = clip_players.query('frame==@t').iloc[0].to_numpy()[2:]
        #
        #     # far_player = clip_players.query('frame==@frame_no and is_far==1').iloc[0].to_numpy()[3:]
        #     # near_player = clip_players.query('frame==@frame_no and is_far==0').iloc[0].to_numpy()[3:]
        #     fx1, fy1, fx2, fy2 = player[:4]
        #     # far_player = np.array([[(fx1 + fx2) / 2, int(fy2)]]).astype(np.float32)[np.newaxis]
        #     # far_player = cv2.perspectiveTransform(far_player, H)[0][0]
        #     # cv2.rectangle(frame, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (255, 0, 0), 3)
        #     nx1, ny1, nx2, ny2 = player[4:]
        #     # near_player = np.array([[(nx1 + nx2) / 2, int(ny2)]]).astype(np.float32)[np.newaxis]
        #     # near_player = cv2.perspectiveTransform(near_player, H)[0][0]
        #     # cv2.rectangle(frame, (int(nx1), int(ny1)), (int(nx2), int(ny2)), (0, 0, 255), 3)
        # else:
        #     # print('Not found', t)
        #     continue

        far_keypoint = clip_skeletons['keypoint'][0, idx]
        far_keypoint_score = clip_skeletons['keypoint_score'][0, idx]

        # near_keypoint = clip_skeletons['keypoint'][1, idx]
        # near_keypoint_score = clip_skeletons['keypoint_score'][1, idx]

        background = frame.copy()
        background[:] = 255

        draw_skeleton(background, far_keypoint, skeleton=COCO_SKELETON, joint_radius=3, bone_thickness=2)
        # draw_skeleton(background, near_keypoint, skeleton=COCO_SKELETON, joint_radius=5, bone_thickness=2)

        # for k, p in enumerate(far_keypoint):
        #     # print(k, p)
        #     cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 0), -1)

        # for k, p in enumerate(near_keypoint):
        #     # print(k, p)
        #     cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 0), -1)

        # background = cv2.resize(background, (1280, 720))
        # frames.append(resized)
        cv2.imshow('background', background)

        # resized = cv2.resize(frame, (1280, 720))
        cv2.imshow('frame', frame)

        if pause:
            k = cv2.waitKey(0)
        else:
            k = cv2.waitKey(5)

        if k == ord(' '):
            pause = not pause
        elif k == ord('q'):
            break
        elif k == ord('x'):
            exit(0)


import torch
import random
import os
import copy
from utilties import preprocess_images, preprocess_maps, preprocess_fixmaps
from config import *

def train_generator(q):
    # data product thread
    while True:
        # dataflow input
        Xims = torch.zeros(video_b_s, num_frames, 3, shape_h, shape_w)
        Ymaps = torch.zeros(video_b_s, num_frames, 1, shape_h_out, shape_w_out)
        Yfixs = torch.zeros(video_b_s, num_frames, 1, shape_h_out, shape_w_out)
        for i in range(0, video_b_s):
            # random sample a dataset
            rd_dataset = ''.join(random.sample(train_dataset, 1))
            rd_videos_train_path = os.path.join(train_dataset_root_path, rd_dataset, train_dir_name)
            videos = list(map(lambda x: os.path.join(rd_videos_train_path, x), os.listdir(rd_videos_train_path)))
            # random sample a sequence
            video_path = ''.join(random.sample(videos, 1))
            img_len = len([video_path + frames_path + f for f in os.listdir(video_path + frames_path) if f.endswith(tuple(img_ext))])
            # imgs need to larger than threshold
            while img_len < least_img_num:
                video_path = ''.join(random.sample(videos, 1))
                img_len = len([video_path + frames_path + f for f in os.listdir(video_path + frames_path) if f.endswith(tuple(img_ext))])
            images = [video_path + frames_path + f for f in os.listdir(video_path + frames_path) if f.endswith(tuple(img_ext))]
            maps = [video_path + maps_path + f for f in os.listdir(video_path + maps_path) if f.endswith(tuple(map_ext))]
            fixs = [video_path + fixs_path + f for f in os.listdir(video_path + fixs_path) if f.endswith(tuple(fix_ext))]
            images.sort()
            maps.sort()
            fixs.sort()
            # random sample a start point
            start = random.randint(0, max(len(images) - num_frames, 0))
            # process dataflow input
            X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_h, shape_w)
            Y = preprocess_maps(maps[start:min(start + num_frames, len(images))], shape_h_out, shape_w_out)
            Y_fix = preprocess_fixmaps(fixs[start:min(start + num_frames, len(images))], shape_h_out, shape_w_out)
            Xims[i, 0:X.shape[0], :, :, :] = copy.deepcopy(X)
            Ymaps[i, 0:Y.shape[0], :, :, :] = copy.deepcopy(Y)
            Yfixs[i, 0:Y_fix.shape[0], :, :, :] = copy.deepcopy(Y_fix)
            # consider for the sequence image number smaller than num_frames
            Xims[i, X.shape[0]:num_frames, :, :, :] = copy.deepcopy(X[-1, :, :, :])
            Ymaps[i, Y.shape[0]:num_frames, :, :, :] = copy.deepcopy(Y[-1, :, :, :])
            Yfixs[i, Y_fix.shape[0]:num_frames, :, :, :] = copy.deepcopy(Y_fix[-1, :, :, :])
        # EnQueue the data
        q.put([Xims, [Ymaps, Yfixs]])
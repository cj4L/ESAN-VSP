import os
import copy
import torch
import datetime
import numpy as np
import scipy
from PIL import Image
from config import *
from utilties import custom_print, preprocess_images, postprocess_predictions
from evaluation import eval_3metric

def val(net, log_txt_file):
    custom_print("starting evaluate", log_txt_file, 'a+')
    custom_print('-' * 100, log_txt_file, 'a+')
    for s in range(len(val_dataset)):
        dataset_name = val_dataset[s]
        dataset_dir = val_dir_name[s]
        val_dataset_path = os.path.join(val_dataset_root_path, dataset_name, dataset_dir)
        videos = list(map(lambda x: os.path.join(val_dataset_path, x), os.listdir(val_dataset_path)))
        videos.sort()
        with torch.no_grad():
            net.eval()
            nss_list, cc_list, sim_list = list(), list(), list()
            for i in range(min(len(videos), 100)):
                images_names = [videos[i] + frames_path + f for f in os.listdir(videos[i] + frames_path) if
                                f.endswith(tuple(img_ext))]
                images_names.sort()
                # for quick eval, each group just use the first 20 imgs
                images_names = images_names[: min(len(images_names), 20)]
                interval = val_num_frames // 2
                images_names_fake = [images_names[0]] * interval + images_names + [images_names[-1]] * interval
                video_results_fake = torch.zeros(len(images_names_fake), shape_h_out, shape_w_out)

                pos = [x for x in range(0, len(images_names_fake))]
                for r in range(val_num_frames):
                    end_point = ((len(images_names_fake) - r) // val_num_frames) * val_num_frames + r
                    pos_group = [pos[x:x + val_num_frames] for x in range(r, end_point, val_num_frames)]
                    for s in range(len(pos_group)):
                        Xims = torch.zeros(1, 5, 3, shape_h, shape_w)
                        X = preprocess_images(images_names_fake[pos_group[s][0]:pos_group[s][-1]+1], shape_h, shape_w)
                        Xims[0] = copy.deepcopy(X)
                        img = Xims.cuda()
                        if s == 0:
                            lstm_forw_state, lstm_back_state = None, None
                        outs, lstm_forw_state, lstm_back_state = net(img, lstm_forw_state, lstm_back_state)
                        video_results_fake[pos_group[s][2]] = outs.cpu().squeeze()

                video_results = video_results_fake[2:-2]

                for k in range(len(images_names)):
                    original_image = Image.open(images_names[k])
                    original_fixation = images_names[k].replace('images', 'fixation/maps')[:-4] + '.mat'
                    original_maps = images_names[k].replace('images', 'maps')
                    fixation = scipy.io.loadmat(original_fixation)['I']
                    maps = np.array(Image.open(original_maps))

                    original_w, original_h = original_image.size
                    post = postprocess_predictions(video_results[k], original_h, original_w)

                    prediction = torch.from_numpy(post).float()
                    fixation = torch.from_numpy(fixation).float()
                    maps = torch.from_numpy(maps).float()
                    nss, cc, sim = eval_3metric(prediction, maps, fixation)
                    nss_list.append(nss)
                    cc_list.append(cc)
                    sim_list.append(sim)

            mean_nss = np.mean(nss_list)
            mean_cc = np.mean(cc_list)
            mean_sim = np.mean(sim_list)

            custom_print(datetime.datetime.now().strftime('%F %T')+' %12s: NSS: [%.4f], CC: [%.4f], sim: [%.4f]'
                         % (dataset_name, mean_nss, mean_cc, mean_sim), log_txt_file, 'a+')

    custom_print('-' * 100, log_txt_file, 'a+')
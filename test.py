import os
import torch
from model import build_model
from utilties import postprocess_predictions, preprocess_images
from config import *
import copy
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpu_id = 'cuda:0'
model_path = './ESAN-epoch_18.pth'

device = torch.device(gpu_id)
videos_test_path = '/home/chenjin/dataset/VisualSaliency/UCF/testing/'
# videos_test_path = '/home/chenjin/dataset/VisualSaliency/DHF1K/testing/'
# videos_test_path = '/home/chenjin/dataset/VisualSaliency/DHF1K/val/'
# videos_test_path = '/home/chenjin/dataset/VisualSaliency/Hollywood-2/testing/'
# videos_test_path = '/home/chenjin/dataset/VisualSaliency/DIEM/testing/'
videos = [videos_test_path + f for f in os.listdir(videos_test_path) if os.path.isdir(videos_test_path + f)]
videos.sort()
nb_videos_test = len(videos)

net = build_model().to(device)
net.load_state_dict(torch.load(model_path, map_location=gpu_id))


with torch.no_grad():
    net.eval()
    for i in range(nb_videos_test):

        output_folder = './results/UCF/' + videos[i].split('/')[-1] + '/'
        # output_folder = './results/DHF1Ktest/' + videos[i].split('/')[-1] + '/'
        # output_folder = './results/DHF1Kval/' + videos[i].split('/')[-1] + '/'
        # output_folder = './results/Hollywood-2/' + videos[i].split('/')[-1] + '/'
        # output_folder = './results/DIEM/' + videos[i].split('/')[-1] + '/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images_names = [videos[i] + frames_path + f for f in os.listdir(videos[i] + frames_path) if
                  f.endswith(('.jpg', '.jpeg', '.png'))]
        images_names.sort()

        interval = val_num_frames // 2
        images_names_fake = [images_names[0]] * interval + images_names + [images_names[-1]] * interval
        video_results_fake = torch.zeros(len(images_names_fake), shape_h_out, shape_w_out)

        pos = [x for x in range(0, len(images_names_fake))]
        for r in range(val_num_frames):
            end_point = ((len(images_names_fake) - r) // val_num_frames) * val_num_frames + r
            pos_group = [pos[x:x + val_num_frames] for x in range(r, end_point, val_num_frames)]
            for s in range(len(pos_group)):
                Xims = torch.zeros(1, 5, 3, shape_h, shape_w)
                X = preprocess_images(images_names_fake[pos_group[s][0]:pos_group[s][-1] + 1], shape_h, shape_w)
                Xims[0] = copy.deepcopy(X)
                img = Xims.to(device)
                if s == 0:
                    lstm_forw_state, lstm_back_state = None, None
                outs, lstm_forw_state, lstm_back_state = net(img, lstm_forw_state, lstm_back_state)
                video_results_fake[pos_group[s][2]] = outs.cpu().squeeze()

        video_results = video_results_fake[2:-2]

        print("Predicting saliency maps for " + videos[i])


        for k in range(len(images_names)):
            original_image = Image.open(images_names[k])
            original_w ,original_h = original_image.size
            post = postprocess_predictions(video_results[k], original_h, original_w)
            result = Image.fromarray(post)
            exact_save_path = os.path.join(output_folder, images_names[k].split('/')[-1][:-4]+'.png')
            result.convert('L').save(exact_save_path)
            # print('a')





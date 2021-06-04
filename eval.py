import torch
import os
from PIL import Image
import scipy.io
import time
import random
import queue
import threading
import numpy as np
import datetime
from tkinter import _flatten

def corr2(map1, map2):
    map1_tmp = map1 - torch.mean(map1)
    map2_tmp = map2 - torch.mean(map2)
    up = torch.sum(map1_tmp * map2_tmp)
    down = torch.sqrt(torch.sum(torch.pow(map1_tmp, 2)) * torch.sum(torch.pow(map2_tmp, 2)))
    return up / (down + 1e-10)

def mean_std_norm(x):
    return (x - torch.mean(x)) / (torch.std(x) + 1e-10)

def min_max_norm(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-10)

def one_zero_norm(x):
    return x / (torch.sum(x) + 1e-10)

def auc_matrix(allthreshes, randfix, Sth, Nfixations, Nfixations_oth):
    tp = torch.zeros(allthreshes.shape[0] + 2, device=device)
    fp = torch.zeros(allthreshes.shape[0] + 2, device=device)
    tp[-1] = 1
    fp[-1] = 1
    Sth = torch.unsqueeze(Sth, dim=1)
    Sth = Sth.expand(Sth.shape[0], len(allthreshes))
    randfix = torch.unsqueeze(randfix, dim=1)
    randfix = randfix.expand(randfix.shape[0], len(allthreshes))
    allthreshes = torch.unsqueeze(allthreshes, dim=0)
    above_tp = torch.sum(Sth > allthreshes, dim=0).float()
    above_fp = torch.sum(randfix > allthreshes, dim=0).float()
    tp[1:-1] = above_tp / (Nfixations + 1e-10)
    fp[1:-1] = above_fp / (Nfixations_oth + 1e-10)
    return torch.trapz(tp, fp).item()

def NSS(P, Q_B):
    up = torch.sum(((P - torch.mean(P)) / (torch.std(P) + 1e-10)) * Q_B)
    down = torch.sum(Q_B)
    return up / (down + 1e-10)

def CC(P, Q_D):
    P = (P - torch.mean(P)) / (torch.std(P) + 1e-10)
    Q_D = (Q_D - torch.mean(Q_D)) / (torch.std(Q_D) + 1e-10)
    return corr2(P, Q_D)

def SIM(P, Q_D):
    P = (P - torch.min(P)) / (torch.max(P) - torch.min(P) + 1e-10)
    P = P / (torch.sum(P) + 1e-10)
    Q_D = (Q_D - torch.min(Q_D)) / (torch.max(Q_D) - torch.min(Q_D) + 1e-10)
    Q_D = Q_D / (torch.sum(Q_D) + 1e-10)
    return torch.sum(torch.min(P, Q_D))

def AUC_Judd(device, P, Q_B):
    jitter = torch.tensor(10000000.0, device=device)
    P = P + (torch.rand(P.shape, device=device) / jitter)
    P = (P - torch.min(P)) / (torch.max(P) - torch.min(P) + 1e-10)
    P = torch.flatten(P)
    Q_B = torch.flatten(Q_B)
    Sth = P[Q_B > 0]
    Nfixations = Sth.shape[0]
    Npixels = P.shape[0]
    allthreshes, _ = torch.sort(Sth, descending=True)
    tp = torch.zeros(Nfixations + 2, device=device)
    fp = torch.zeros(Nfixations + 2, device=device)
    tp[-1] = 1
    fp[-1] = 1
    P = torch.unsqueeze(P, dim=1)
    P_matrix = P.expand(P.shape[0], Nfixations)
    allthreshes_matrix = torch.unsqueeze(allthreshes, dim=0)
    aboveth_matrix = torch.sum(P_matrix > allthreshes_matrix, dim=0).float()
    tp[1:-1] = torch.arange(1, Nfixations + 1, device=device, dtype=torch.float) / (Nfixations + 1e-10)
    fp[1:-1] = (aboveth_matrix - torch.arange(1, Nfixations + 1, device=device, dtype=torch.float)) / (Npixels - Nfixations + 1e-10)
    return torch.trapz(tp, fp).item()

def AUC_shuffle(device, P, Q_B, other):
    Nsplits = 100
    P = (P - torch.min(P)) / (torch.max(P) - torch.min(P) + 1e-10)
    P = torch.flatten(P)
    Q_B = torch.flatten(Q_B)
    Oth = torch.flatten(other)
    Sth = P[Q_B > 0]
    Nfixations = Sth.shape[0]
    ind = torch.nonzero(Oth).squeeze()
    Nfixations_oth = min(Nfixations, ind.shape[0])
    ind_matrix = [ind] * Nsplits
    randfix = list(map(lambda x: P[np.random.choice(x.cpu().numpy(), Nfixations_oth)], ind_matrix))
    randfix_matrix = torch.stack(randfix[:], dim=0)
    max_matrix = list(torch.max(torch.cat((Sth.expand_as(randfix_matrix), randfix_matrix), dim=1), dim=1)[0].cpu().numpy())
    allthreshes = list(map(lambda x: torch.sort(torch.arange(0, x, 0.1, device=device), descending=True)[0], max_matrix))
    Sth_list = [Sth] * Nsplits
    Nfixations_list = [Nfixations] * Nsplits
    Nfixations_oth_list = [Nfixations_oth] * Nsplits
    auc_list = list(map(auc_matrix, allthreshes, randfix, Sth_list, Nfixations_list, Nfixations_oth_list))
    return np.mean(auc_list)

def eval_all_5metric(device, P, Q_D, Q_B, other):
    mean_std_norm_P = mean_std_norm(P)
    mean_std_norm_Q_D = mean_std_norm(Q_D)
    min_max_norm_P = min_max_norm(P)
    min_max_norm_Q_D = min_max_norm(Q_D)
    sum_Q_B = torch.sum(Q_B)
    nss = (torch.sum(mean_std_norm_P * Q_B) / (sum_Q_B + 1e-10)).item()
    cc = corr2(mean_std_norm_P, mean_std_norm_Q_D).item()
    sim = torch.sum(torch.min(one_zero_norm(min_max_norm_P), one_zero_norm(min_max_norm_Q_D))).item()
    aucj = AUC_Judd(device, P, Q_B)
    aucs = AUC_shuffle(device, P, Q_B, other)
    return nss, cc, sim, aucj, aucs


def process_othermap(otherpath, oripath):
    Ix = scipy.io.loadmat(otherpath)['I']
    cur_h, cur_w = Ix.shape
    ori = scipy.io.loadmat(oripath)['I']
    ori_h, ori_w = ori.shape
    rescale = np.array([[ori_w / cur_w], [ori_h / cur_h]])
    pos = np.vstack(np.nonzero(Ix))
    pos = np.round(pos * rescale).astype(np.int16)
    condition = (pos[0] >= 0) & (pos[0] < ori_h) & (pos[1] >= 0) & (pos[1] < ori_w)
    newpos = pos[:, condition]
    return newpos


def gen_test_data(q, result_path, gt_path):
    tmp = list()
    sequence_path = os.listdir(result_path)
    for s in range(len(sequence_path)):
        prediction_path = os.listdir(os.path.join(result_path, sequence_path[s]))
        tmp.append(list(map(lambda x: os.path.join(sequence_path[s], x), prediction_path)))
    all_test_info_list = list(_flatten(tmp))
    all_results_list = list(map(lambda x: os.path.join(result_path, x), all_test_info_list))
    all_gt_fixations_list = list(map(lambda x: os.path.join(gt_path, x.split('/')[0], 'fixation/maps', x.split('/')[1][:-4] + '.mat'), all_test_info_list))
    all_gt_maps_list = list(map(lambda x: os.path.join(gt_path, x.split('/')[0], 'maps', x.split('/')[1]), all_test_info_list))

    for i in range(len(all_test_info_list)):
        prediction = np.array(Image.open(all_results_list[i]).resize(Image.open(all_gt_maps_list[i]).size, Image.BILINEAR))
        fixation = scipy.io.loadmat(all_gt_fixations_list[i])['I']
        maps = np.array(Image.open(all_gt_maps_list[i]))
        otherMap = np.zeros_like(fixation)

        random_ids = random.sample(range(len(all_test_info_list)), 10)
        random_fixations_list = list(map(lambda x: all_gt_fixations_list[x], random_ids))
        random_match_list = [all_gt_fixations_list[i]] * 10

        random_fixations = list(map(process_othermap, random_fixations_list, random_match_list))
        final_pos = np.hstack(list(_flatten(random_fixations)))
        otherMap[final_pos[0], final_pos[1]] = 1

        q.put([prediction, fixation, maps, otherMap])


def test():
    print(datetime.datetime.now().strftime('%F %T'), ' now is evaluating')
    nss_list = list()
    cc_list = list()
    sim_list = list()
    aucj_list = list()
    aucs_list = list()
    num = 1

    for i in range(all_test_length):
        prediction, fixation, maps, otherMap = q.get()
        prediction = torch.from_numpy(prediction).float().to(device)
        fixation = torch.from_numpy(fixation).float().to(device)
        maps = torch.from_numpy(maps).float().to(device)
        otherMap = torch.from_numpy(otherMap).float().to(device)

        nss, cc, sim, aucj, aucs = eval_all_5metric(device, prediction, maps, fixation, otherMap)

        nss_list.append(nss)
        cc_list.append(cc)
        sim_list.append(sim)
        aucj_list.append(aucj)
        aucs_list.append(aucs)
        num += 1
        if (num) % 1000 == 0:
            print(datetime.datetime.now().strftime('%F %T'), num)

    N = torch.mean(torch.tensor(nss_list))
    c = torch.mean(torch.tensor(cc_list))
    s = torch.mean(torch.tensor(sim_list))
    j = torch.mean(torch.tensor(aucj_list))
    ss = torch.mean(torch.tensor(aucs_list))

    print(datetime.datetime.now().strftime('%F %T'), 'done')
    print(j, s, ss, c, N)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    result_root_path = './results/UCF'
    gt_root_path = '/home/chenjin/dataset/VisualSaliency/UCF/testing'
    # result_root_path = './results/DHF1Kval'
    # gt_root_path = '/home/chenjin/dataset/VisualSaliency/DHF1K/val'
    # result_root_path = './results/Hollywood-2'
    # gt_root_path = '/home/chenjin/dataset/VisualSaliency/Hollywood-2/testing'
    # result_root_path = './results/DIEM'
    # gt_root_path = '/home/chenjin/dataset/VisualSaliency/DIEM/testing'

    tmp = list()
    sequence_path = os.listdir(result_root_path)
    for s in range(len(sequence_path)):
        prediction_path = os.listdir(os.path.join(result_root_path, sequence_path[s]))
        tmp.append(list(map(lambda x: os.path.join(sequence_path[s], x), prediction_path)))
    all_test_length = len(list(_flatten(tmp)))

    gpu_id = 'cuda:0'
    device = torch.device(gpu_id)
    batch_size = 1
    q = queue.Queue(maxsize=40)
    p1 = threading.Thread(target=gen_test_data, args=(q, result_root_path, gt_root_path))
    c1 = threading.Thread(target=test)

    p1.start()
    time.sleep(1)
    c1.start()







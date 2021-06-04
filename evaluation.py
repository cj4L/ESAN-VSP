import torch

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

def eval_3metric(P, Q_D, Q_B):
    mean_std_norm_P = mean_std_norm(P)
    mean_std_norm_Q_D = mean_std_norm(Q_D)
    min_max_norm_P = min_max_norm(P)
    min_max_norm_Q_D = min_max_norm(Q_D)
    sum_Q_B = torch.sum(Q_B)
    nss = (torch.sum(mean_std_norm_P * Q_B) / (sum_Q_B + 1e-10)).item()
    cc = corr2(mean_std_norm_P, mean_std_norm_Q_D).item()
    sim = torch.sum(torch.min(one_zero_norm(min_max_norm_P), one_zero_norm(min_max_norm_Q_D))).item()
    return nss, cc, sim









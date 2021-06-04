import torch
from torch import nn

class kl_divergence(nn.Module):
    def __init__(self):
        super(kl_divergence, self).__init__()

    def forward(self, y_true, y_pred):
        max_y_pred = torch.max(y_pred)
        y_pred = y_pred / max_y_pred
        y_bool = (y_true > 0.1).float()
        sum_y_true = torch.sum(y_true)
        sum_y_pred = torch.sum(y_pred)
        y_true = y_true / (sum_y_true + 1e-10)
        y_pred = y_pred / (sum_y_pred + 1e-10)
        loss = torch.sum(y_bool * y_true * torch.log((y_true / (y_pred + 1e-10) + 1e-10)))
        return loss

class correlation_coefficient(nn.Module):
    def __init__(self):
        super(correlation_coefficient, self).__init__()

    def forward(self, y_true, y_pred):
        max_y_pred = torch.max(y_pred)
        y_pred = y_pred / max_y_pred
        y_bool = (torch.max(y_true) > 0.1).float()
        sum_y_true = torch.sum(y_true)
        sum_y_pred = torch.sum(y_pred)
        y_true = y_true / (sum_y_true + 1e-10)
        y_pred = y_pred / (sum_y_pred + 1e-10)
        N = y_pred.shape[0] * y_pred.shape[1]
        sum_prod = torch.sum(y_true * y_pred)
        sum_x = torch.sum(y_true)
        sum_y = torch.sum(y_pred)
        sum_x_square = torch.sum(y_true.pow(2)) + 1e-10
        sum_y_square = torch.sum(y_pred.pow(2)) + 1e-10
        num = sum_prod - ((sum_x * sum_y) / N)
        den = torch.sqrt((sum_x_square - sum_x.pow(2) / N) * (sum_y_square - sum_y.pow(2) / N))
        loss = torch.sum(y_bool * (-2 * num/den))
        return loss

class nss(nn.Module):
    def __init__(self):
        super(nss, self).__init__()

    def forward(self, y_true, y_pred):
        max_y_pred = torch.max(y_pred)
        y_pred = y_pred / max_y_pred
        y_bool = (torch.max(y_true) > 0.1).float()
        y_mean = torch.mean(y_pred)
        y_std = torch.std(y_pred)
        y_pred = (y_pred - y_mean) / (y_std + 1e-10)
        loss = -1 * torch.sum(y_bool * (torch.sum(y_true * y_pred)) / torch.sum(y_true))
        return loss

class sim(nn.Module):
    def __init__(self):
        super(sim, self).__init__()

    def forward(self, y_true, y_pred):
        P = (y_pred - torch.min(y_pred)) / (torch.max(y_pred) - torch.min(y_pred) + 1e-10)
        P = P / (torch.sum(P) + 1e-10)
        Q_D = (y_true - torch.min(y_true)) / (torch.max(y_true) - torch.min(y_true) + 1e-10)
        Q_D = Q_D / (torch.sum(Q_D) + 1e-10)
        loss = torch.sum(torch.min(P, Q_D))
        return loss



class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_kl = kl_divergence()
        self.loss_cc = correlation_coefficient()
        self.loss_nss = nss()
        self.loss_sim = sim()

    def forward(self, y_true_map, y_true_fix, y_pred):
        loss1 = self.loss_kl(y_true_map, y_pred)
        loss2 = self.loss_cc(y_true_map, y_pred)
        loss3 = self.loss_nss(y_true_fix, y_pred)
        loss4 = self.loss_sim(y_true_map, y_pred)

        loss = loss1 + loss2 + loss3 + loss4
        return loss


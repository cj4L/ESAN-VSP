from loss import Loss
from torch.optim import Adam
from utilties import custom_print
import datetime
import torch
from val import val

def train(net, q, log_txt_file, lr, lr_de, epochs, iters, log_interval, val_interval, model_save_name):

    loss = Loss().cuda()
    optimizer = Adam(net.parameters(), lr, weight_decay=1e-6)
    for e in range(epochs):
        ave_video_loss = 0
        for d in range(iters):
            img, label = q.get()
            img = img.cuda()
            for k in range(len(label)):
                label[k] = label[k].cuda()

            outs, _, _ = net(img)
            all_loss = 0
            o_B, o_N, _, _, _ = outs.size()
            for i in range(o_B):
                for j in range(o_N):
                    cur_out = outs[i, j, 0, :, :]
                    cur_map_label = label[0][i, j * 5 + 2, 0]
                    cur_fix_label = label[1][i, j * 5 + 2, 0]
                    cur_loss = loss(cur_map_label, cur_fix_label, cur_out)
                    all_loss += cur_loss
            all_video_loss = all_loss / o_B / o_N
            all_video_loss.backward()
            ave_video_loss += all_video_loss.item()

            optimizer.step()

            if (d + 1) % log_interval == 0:
                custom_print(datetime.datetime.now().strftime('%F %T') +
                             ' lr: %e, epoch: [%d/%d], iter: [%d/%d], vloss: [%.4f]' %
                             (lr, e, epochs, d + 1, iters, ave_video_loss / log_interval), log_txt_file, 'a+')
                ave_video_loss = 0

        if (e + 1) % val_interval == 0:
            net.eval()
            val(net, log_txt_file)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), model_save_name + '_epoch_%d.pth' % (e))
            else:
                torch.save(net.state_dict(), model_save_name + '_epoch_%d.pth' % (e))
            net.train()

        if (e + 1) in lr_de:
            lr = lr / 2
            optimizer = Adam(net.parameters(), lr, weight_decay=1e-6)
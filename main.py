import os
import queue
import threading
import torch
from model import build_model, weights_init
from utilties import custom_print
from train import train
from train_data_producer import train_generator
torch.backends.cudnn.benchmark = True




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # projects setting
    project_name = 'ESAN'
    vgg_path = '/home/chenjin/weights/vgg16_feat.pth'
    model_path = './models/ESAN-epoch_18.pth'


    lr = 1e-7
    lr_de = [100]
    epochs = 20
    iters = 1000
    log_interval = 100
    val_interval = 1
    continue_flag = False


    # create log dir
    log_root = './logs'
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    # create log txt
    log_txt_file = os.path.join(log_root, project_name + '_log.txt')
    custom_print(project_name, log_txt_file, 'w')

    # create model save dir
    models_root = './models'
    if not os.path.exists(models_root):
        os.makedirs(models_root)

    model_save_name = os.path.join(models_root, project_name)

    # build net
    net = build_model()
    net.apply(weights_init)



    if not continue_flag:
        net.base.load_state_dict(torch.load(vgg_path))
    else:
        net.load_state_dict(torch.load(model_path), strict=False)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net.cuda(), device_ids=[0, 1])
    elif torch.cuda.is_available():
        net = net.cuda()

    # data queue
    q = queue.Queue(maxsize=40)

    # data thread
    t1 = threading.Thread(target=train_generator, args=(q, ))
    t2 = threading.Thread(target=train_generator, args=(q, ))

    # train thread
    c1 = threading.Thread(target=train, args=(net, q, log_txt_file, lr, lr_de, epochs, iters, log_interval, val_interval, model_save_name))

    t1.start()
    t2.start()
    c1.start()
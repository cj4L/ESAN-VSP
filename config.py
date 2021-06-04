
# parameters
# batch size
video_b_s = 2*2
# frames num
num_frames = 15
# input size
shape_w = 320
shape_h = 256
# output size
shape_w_out = 160
shape_h_out = 128


# train dataset parameters
train_dataset_root_path = '/home/chenjin/dataset/VisualSaliency'
train_dataset = ['DHF1K', 'Hollywood-2', 'UCF']
# train_dataset = ['Hollywood-2']
train_dir_name = 'training'

maps_path = '/maps/'
fixs_path = '/fixation/maps/'
frames_path = '/images/'

img_ext = ['.jpg', '.jpeg', '.png']
map_ext = ['.jpg', '.jpeg', '.png']
fix_ext = ['.mat']
least_img_num = 10


# val dataset parameters
val_dataset_root_path = '/home/chenjin/dataset/VisualSaliency'
val_dataset = ['DHF1K', 'Hollywood-2', 'UCF', 'DIEM']
val_dir_name = ['val', 'testing', 'testing', 'testing']
val_num_frames = 5




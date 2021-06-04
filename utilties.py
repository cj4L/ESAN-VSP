from __future__ import division
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import scipy.io
import torch.nn.functional as F
import scipy.ndimage
# from scipy.misc import imread, imresize

def img_transformers(shape_h, shape_w):
    img_transform = transforms.Compose([transforms.Resize((shape_h, shape_w)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return img_transform

def map_transformers(shape_h, shape_w):
    map_transform = transforms.Compose([transforms.Resize((shape_h, shape_w)), transforms.ToTensor()])
    return map_transform

def extra_transformers(shape_h, shape_w):
    extra_transform = transforms.Compose([transforms.Resize((shape_h, shape_w)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.449], std=[0.226])])
    return extra_transform


def padding(img, shape_h=240, shape_w=320, channels=3):
    img_padded = torch.zeros(channels, shape_h, shape_w)
    # if channels == 1:
    #     img_padded = torch.zeros(shape_h, shape_w)

    original_shape = img.size
    rows_rate = original_shape[1]/shape_h
    cols_rate = original_shape[0]/shape_w

    if rows_rate > cols_rate:
        new_cols = (original_shape[0] * shape_h) // original_shape[1]
        if channels == 3:
            trans = img_transformers(shape_h, new_cols)
        else:
            trans = map_transformers(shape_h, new_cols)
        img = trans(img)
        if new_cols > shape_w:
            new_cols = shape_w
        img_padded[:, :, ((img_padded.shape[2] - new_cols) // 2):((img_padded.shape[2] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[1] * shape_w) // original_shape[0]
        if channels == 3:
            trans = img_transformers(new_rows, shape_w)
        else:
            trans = map_transformers(new_rows, shape_w)
        img = trans(img)
        if new_rows > shape_h:
            new_rows = shape_h
        img_padded[:, ((img_padded.shape[1] - new_rows) // 2):((img_padded.shape[1] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def extra_padding(img, shape_h=240, shape_w=320, channels=1):
    img_padded = torch.zeros(channels, shape_h, shape_w)

    original_shape = img.size
    rows_rate = original_shape[1]/shape_h
    cols_rate = original_shape[0]/shape_w

    if rows_rate > cols_rate:
        new_cols = (original_shape[0] * shape_h) // original_shape[1]
        trans = extra_transformers(shape_h, new_cols)
        img = trans(img)
        if new_cols > shape_w:
            new_cols = shape_w
        img_padded[:, :, ((img_padded.shape[2] - new_cols) // 2):((img_padded.shape[2] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[1] * shape_w) // original_shape[0]
        trans = extra_transformers(new_rows, shape_w)
        img = trans(img)
        if new_rows > shape_h:
            new_rows = shape_h
        img_padded[:, ((img_padded.shape[1] - new_rows) // 2):((img_padded.shape[1] - new_rows) // 2 + new_rows), :] = img

    return img_padded

# def padding(img, shape_h=240, shape_w=320, channels=3):
#     img_padded = torch.zeros(channels, shape_h, shape_w)
#     if channels == 1:
#         img_padded = torch.zeros(shape_h, shape_w)
#
#     original_shape = img.size
#     rows_rate = original_shape[1]/shape_h
#     cols_rate = original_shape[0]/shape_w
#
#     if rows_rate > cols_rate:
#         new_cols = (original_shape[0] * shape_h) // original_shape[1]
#
#         trans = transformers(shape_h, new_cols)
#
#         # img = img.resize((shape_h, new_cols), Image.BILINEAR)
#         # img = imresize(img, (new_rows,shape_c))
#         f = img
#         f.save('f.jpg')
#
#         img = trans(img)
#
#         d = img.permute(1, 2, 0).numpy()
#         d = d * 255
#         d = d.astype(np.uint8)
#         result = Image.fromarray(d)
#         result.save('d.jpg')
#
#         # img = imresize(img, (shape_r, new_cols))
#         if new_cols > shape_w:
#             new_cols = shape_w
#         img_padded[:, :, ((img_padded.shape[2] - new_cols) // 2):((img_padded.shape[2] - new_cols) // 2 + new_cols)] = img
#         c = img_padded.permute(1, 2, 0).numpy()
#         c = c*255
#         c = c.astype(np.uint8)
#         result = Image.fromarray(c)
#         result.save('a.jpg')
#
#
#
#     else:
#         new_rows = (original_shape[1] * shape_w) // original_shape[0]
#
#         trans = transformers(new_rows, shape_w)
#
#         # img = img.resize((shape_w, new_rows), Image.BILINEAR)
#         # img = imresize(img, (new_rows,shape_c))
#         # f = img
#         # f.save('f.jpg')
#
#
#         img = trans(img)
#         # d = img.permute(1, 2, 0).numpy()
#         # d = d * 255
#         # d = d.astype(np.uint8)
#         # result = Image.fromarray(d)
#         # result.save('d.jpg')
#
#
#         if new_rows > shape_h:
#             new_rows = shape_h
#         # a = ((img_padded.shape[1] - new_rows) // 2)
#         # b = ((img_padded.shape[1] - new_rows) // 2 + new_rows)
#         img_padded[:, ((img_padded.shape[1] - new_rows) // 2):((img_padded.shape[1] - new_rows) // 2 + new_rows), :] = img
#         # c = img_padded.permute(1, 2, 0).numpy()
#         # c = c*255
#         # c = c.astype(np.uint8)
#         # result = Image.fromarray(c)
#         # result.save('a.jpg')
#
#     return img_padded


def resize_fixation(img, rows=480, cols=640):
    out = torch.zeros(rows, cols)
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out


def padding_fixation(img, shape_h=480, shape_w=640):
    img_padded = torch.zeros(shape_h, shape_w)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_h
    cols_rate = original_shape[1]/shape_w

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_h) // original_shape[0]
        img = resize_fixation(img, rows=shape_h, cols=new_cols)
        if new_cols > shape_w:
            new_cols = shape_w
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols),] = img
    else:
        new_rows = (original_shape[0] * shape_w) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_w)
        if new_rows > shape_h:
            new_rows = shape_h
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def preprocess_images(paths, shape_h, shape_w):
    ims = torch.zeros(len(paths), 3, shape_h, shape_w)

    # aaa = 'banana_035.jpg'
    # original_image = Image.open(aaa)
    # padded_image = padding(original_image, shape_h, shape_w, 3)

    for i, path in enumerate(paths):
        # original_image = cv2.imread(path)
        # original_image = mpimg.imread(path)
        original_image = Image.open(path)
        if original_image.mode == 'RGB':
        # if original_image.ndim == 2:
        #     copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
        #     copy[:, :, 0] = original_image
        #     copy[:, :, 1] = original_image
        #     copy[:, :, 2] = original_image
        #     original_image = copy
            padded_image = padding(original_image, shape_h, shape_w, 3)
            ims[i] = padded_image
        else:
            padded_image = extra_padding(original_image, shape_h, shape_w, 1)
            ims[i] = padded_image

    # ims[:, :, :, 0] -= 103.939
    # ims[:, :, :, 1] -= 116.779
    # ims[:, :, :, 2] -= 123.68
    # ims = ims[:, :, :, ::-1]
    # ims = ims.transpose((0, 3, 1, 2))

    return ims


def preprocess_maps(paths, shape_h, shape_w):
    ims = torch.zeros(len(paths), 1, shape_h, shape_w)

    for i, path in enumerate(paths):
        # original_map = cv2.imread(path, 0)
        # original_map = mpimg.imread(path)
        # original_map = imread(path)
        # f = Image.open(path)
        # f.save('f.jpg')
        original_map = Image.open(path).convert('L')
        padded_map = padding(original_map, shape_h, shape_w, 1)
        # d = padded_map.squeeze().numpy()
        # d = d * 255
        # d = d.astype(np.uint8)
        # result = Image.fromarray(d)
        # result.save('d.jpg')


        ims[i, 0, :, :] = padded_map
    # aaa = ims.numpy()
        # ims[i, :, :, 0] /= 255.0

    return ims


def preprocess_fixmaps(paths, shape_h, shape_w):
    ims = torch.zeros(len(paths), 1, shape_h, shape_w)

    for i, path in enumerate(paths):
        tmp = scipy.io.loadmat(path)
        # try:
        fix_map = tmp['I']
        # except Exception:
        #     print(path)
        ims[i, 0, :, :] = padding_fixation(fix_map, shape_h=shape_h, shape_w=shape_w)

    return ims


def postprocess_predictions(pred, shape_h, shape_w):
    predictions_shape = pred.shape
    rows_rate = shape_h / predictions_shape[0]
    cols_rate = shape_w / predictions_shape[1]

    pred = pred / torch.max(pred) * 255

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_h) // predictions_shape[0]
        # pred = cv2.resize(pred, (new_cols, shape_r))
        pred = F.interpolate(pred.unsqueeze(dim=0).unsqueeze(dim=0), [shape_h, new_cols], mode='bilinear').squeeze()
        img = pred[:, ((pred.shape[1] - shape_w) // 2):((pred.shape[1] - shape_w) // 2 + shape_w)]
    else:
        new_rows = (predictions_shape[0] * shape_w) // predictions_shape[1]
        # pred = cv2.resize(pred, (shape_c, new_rows))
        pred = F.interpolate(pred.unsqueeze(dim=0).unsqueeze(dim=0), [new_rows, shape_w], mode='bilinear').squeeze()
        # pred = imresize(pred, (new_rows, shape_c))
        img = pred[((pred.shape[0] - shape_h) // 2):((pred.shape[0] - shape_h) // 2 + shape_h), :]

    # aaa = img.cpu().detach().numpy()

    img = scipy.ndimage.filters.gaussian_filter(img.cpu().detach().numpy(), sigma=7)
    img = img / np.max(img) * 255

    return img

def custom_print(context, log_file, mode):
    #custom print and log out function
    if mode == 'w':
        fp = open(log_file, mode)
        fp.write(context + '\n')
        fp.close()
    elif mode == 'a+':
        print(context)
        fp = open(log_file, mode)
        print(context, file=fp)
        fp.close()
    else:
        raise Exception('other file operation is unimplemented !')
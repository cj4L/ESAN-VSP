# Video Saliency Prediction Using Enhanced Spatiotemporal Alignment Network
* PDF: [arXiv](https://arxiv.org/abs/2001.00292) or [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0031320320304180)

## Overview of our method
![](https://github.com/cj4L/ESAN-VSP/raw/master/pic/network.png)

## Results download
&emsp;Our results on DHF1K, HollyWood-2, UCF and DIEM can be downloaded by Google Drive or Baidu wangpan. Which consists of 4 training settings with the training set(s) from (i)DHF1K, (ii)HollyWood-2, (iii)UCF sports, (iv)DHF1K + HollyWood-2 + UCF sports.

  settings/datasets  |DHF1K | HollyWood-2|UCF|DIEM|
  ---| ---  | ---   | ---   | ---   | 
  setting(i)|<a href="https://drive.google.com/open?id=1TRheAJrYT4KxZSeO7NCFLW0KgRXwv4vg"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=1L0hbBpC9OoFGXCg-OG24l-DthKqnBIKA"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=1mpwCQdQRX0ZTqoJzMdDNvAfJElPFCfwp"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=1qCZ2gsiC085datnSKIKMvNhZr-pNv69v"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a>
  setting(ii)|<a href="https://drive.google.com/open?id=1-zyWZhhmPvG8oo_z7nD-qznr2keNRgCh"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=10zDWXK-ng4BaBNjbIeS6LefF8QYGj_Rt"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=1XlKBv7oukaUM2BDQKrjg9UTYjFEObOqX"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=1DCYzK1SQ9AWYq0arNRt0BEHnc3aLhs5k"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a>
  setting(iii)|<a href="https://drive.google.com/open?id=13CxZXPatYP2O7KR2hQA9NPqp-Dc63nJy"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=1XrAogBffOsEdh3x7aB-Vb5cPjnGy_f7k"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=1KN6enpI3P8LvQtN7CNYe21uYTNZaMqJF"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=1Sd1kFHA7NRUVI-hf_X66VW7Yx2V08wvs"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a>
  setting(iv)|<a href="https://drive.google.com/open?id=10zYqjO2KyEe0tZ-K4iFrtcyVt0Q0Irc3"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=1AS7Zhz7shui2EHeL1srEhGgpLfJfeo2u"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=1XkLKAlUuCl8tgdXFfF_9JA6AqXpEb4nw"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a> | <a href="https://drive.google.com/open?id=12ktmGBcjb2EkYMEfCKfA21T7Mn1IYoJs"><img src="https://github.com/cj4L/ESAN-VSP/raw/master/pic/googledrive.png" width="35" height="30"></a>
  
## Preparation
### Datasets download
&emsp;How to get million pictures is the first barrier in video saliency prediction task. Thanks to [@wenguanwang](https://github.com/wenguanwang) proposed, pre-processed and published some pictures. DHF1K, HollyWood-2, UCF can be downloaded by Google Drive in his [repo](https://github.com/wenguanwang/DHF1K#dhf1k).

&emsp;For convenience, we clone and upload the duplicate of HollyWood-2 to Baidu wangpan to download. All data belongs to the original author, sharing the duplicate is only for academic development, if there is any infringement, please contact me.

&emsp;We adopt [here](https://github.com/wenguanwang/DHF1K/blob/master/make_gauss_masks.m) to pre-processed the DIEM datasets. Following
[STRA-Net](https://github.com/ashleylqx/STRA-Net), the testing sets contain 20 selected videos which including first 300 frames per video, and some frames without labels are eliminated. Click [here](https://drive.google.com/open?id=1rCvtBQxMdqoy9gmisZzhOi_LLBddFArk) to download in Google Drive.

### Models download
&emsp;We use the VGG16 pretrained weights from PyTorch official version [here](https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg16), and we remove the last few layers. Click [here](https://drive.google.com/open?id=1Ar3pF4bzNWX-CSXaWcQqSoTRS_46KSLl) to download in Google Drive, click [here](https://pan.baidu.com/s/1uJFG2O3_Vc6qdFhlsGrDhQ) to download in Baidu wangpan.

&emsp;The trained model in setting(iv): click [here](https://drive.google.com/open?id=1sJHoD-2ypsLzyKSn3JiHl2pdSmwpakpc) to download in Google Drive, click [here](https://pan.baidu.com/s/1Lqwu-LYIrO1JgoxQkcnaBw) to download in Baidu wangpan.

### Experiments platform
&emsp;OS: ubuntu 16.04

&emsp;RAM: 64G

&emsp;CPU: Intel i7-8700

&emsp;GPU: Nvidia RTX 2080Ti * 2

&emsp;Language: Python 3

### Enviroment dependencies
&emsp;Due to the compilation of DCN need earlier version PyTorch and torch.trapz() function need newer, so our dependencies are listed:

&emsp;Training and testing phase: PyTorch 1.0.1.post2, torchvision, Pillow, numpy, scipy and other dependencies.

&emsp;Eval phase: PyTorch 1.2 or newer, Pillow, numpy, scipy, tkinter and other dependencies.

## First
* We use [DCN-V2](https://github.com/CharlesShang/DCNv2) and modify something in dcn_v2.py, you need replace file and re-compile it. 

## Test
* Get or download the dataset.
* Download pretrained model in [Google Drive](https://drive.google.com/file/d/1sJHoD-2ypsLzyKSn3JiHl2pdSmwpakpc/view?usp=sharing).
* Modify the config.py and run test.py.

## Val
* Modify the config.py and run eval.py.

## Train
* Get or download the dataset.
* Download VGG16 pretrained weights in [Google Drive](https://drive.google.com/file/d/1KIWIspVxLRwv8bzOuMn6lY8kStoedToV/view?usp=sharing). Actually is from PyTorch offical model weights, expect for deleting the last serveral layers.
* Modify the config.py and run main.py.

### Notes
* There is something wrong about the share of BaiduPan, contact me if want.

#### Schedule
- [x] Create github repo (2019.12.29)
- [x] Release arXiv pdf (2020.01.05)
- [x] Release all results (2020.01.09)
- [x] Add preparation (2020.01.13)
- [x] Test and Train code (2021.06.04)

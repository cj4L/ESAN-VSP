import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from convlstm import ConvLSTM
from dcn_v2 import DCN

# vgg choice
base = {'vgg': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

# vgg16
def vgg16(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

# something originate from EDVR(https://github.com/xinntao/EDVR)
class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=512, groups=8):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=groups)

        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat((nbr_fea_l[2], ref_fea_l[2]), dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(L3_offset, nbr_fea_l[2]))
        # L2
        L2_offset = torch.cat((nbr_fea_l[1], ref_fea_l[1]), dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat((L2_offset, L3_offset * 2), dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack(L2_offset, nbr_fea_l[1])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat((L2_fea, L3_fea), dim=1)))
        # L1
        L1_offset = torch.cat((nbr_fea_l[0], ref_fea_l[0]), dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat((L1_offset, L2_offset * 2), dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack(L1_offset, nbr_fea_l[0])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat((L1_fea, L2_fea), dim=1))
        # Cascading
        offset = torch.cat((L1_fea, ref_fea_l[0]), dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(offset, L1_fea))

        return L1_fea


class Model(nn.Module):
    def __init__(self, base):
        super(Model, self).__init__()
        self.center = 5 // 2
        self.base = nn.ModuleList(base)
        self.extract = [16, 23, 30]
        self.nf = 512
        self.groups = 8
        self.pcd_align = PCD_Align(nf=self.nf, groups=self.groups)
        self.final_conv = nn.Sequential(nn.Conv2d(512, 1, 3, 1, 1))
        self.final_sigmoid = nn.Sequential(nn.Sigmoid())
        self.att_sigmoid = nn.Sequential(nn.Sigmoid())
        self.L1_conv = nn.Sequential(nn.Conv2d(256, 512, 1, 1))
        self.lstm_forw = ConvLSTM(input_size=(32, 40), input_dim=512, hidden_dim=[512], kernel_size=(3, 3),
                             padding=1, dilation=1, num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.lstm_back = ConvLSTM(input_size=(32, 40), input_dim=512, hidden_dim=[512], kernel_size=(3, 3),
                             padding=1, dilation=1, num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.sc_fusion = nn.Conv2d(5 * self.nf, self.nf, 1, 1, bias=True)


    def forward(self, x, last_forw_state=None, last_back_state=None):
        B, N, C, H, W = x.size()
        # (B,N,C,H,W) --> (BN,C,H,W)
        x = torch.cat(torch.split(x, split_size_or_sections=1, dim=0), dim=1).squeeze()

        ## extract vgg p3, p4, p5 features
        p = list()
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                p.append(x)

        # (BN,C,H,W) --> (B,N,C,H,W)
        L1_fea = torch.stack(torch.split(self.L1_conv(p[0]), split_size_or_sections=N, dim=0), dim=0)
        L2_fea = torch.stack(torch.split(p[1], split_size_or_sections=N, dim=0), dim=0)
        L3_fea = torch.stack(torch.split(p[2], split_size_or_sections=N, dim=0), dim=0)

        # (B,N,C,H,W) --> (B,5,C,H,W)*group
        L1_fea_split = torch.split(L1_fea, split_size_or_sections=5, dim=1)
        L2_fea_split = torch.split(L2_fea, split_size_or_sections=5, dim=1)
        L3_fea_split = torch.split(L3_fea, split_size_or_sections=5, dim=1)

        ## pcd group align
        # (B,5,C,H,W)*group --> (B,N/5,C,H,W)
        aligned_fea_group = []
        for gr in range(len(L1_fea_split)):
            ref_fea_l = [
                L1_fea_split[gr][:, self.center, :, :, :].clone(), L2_fea_split[gr][:, self.center, :, :, :].clone(),
                L3_fea_split[gr][:, self.center, :, :, :].clone()
            ]
            aligned_fea = []
            for sp in range(5):
                nbr_fea_l = [
                    L1_fea_split[gr][:, sp, :, :, :].clone(), L2_fea_split[gr][:, sp, :, :, :].clone(),
                    L3_fea_split[gr][:, sp, :, :, :].clone()
                ]
                aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
            aligned_fea_result = torch.cat(aligned_fea, dim=1)
            fused_fea = self.sc_fusion(aligned_fea_result)
            aligned_fea_group.append(fused_fea)
        final_aligned_fea_group = torch.stack(aligned_fea_group, dim=1)
        reversed_feature = torch.stack(aligned_fea_group[::-1], dim=1)

        ## Bi convlstm
        # (B,N/5,C,H,W) --> (B,N/5,C,H,W)

        # forward convlstm
        lstm_forw_output, lstm_forw_output_last_state = self.lstm_forw(final_aligned_fea_group, last_forw_state)
        lstm_forw_output = lstm_forw_output[0]

        # backforw convlstm
        lstm_back_output, lstm_back_output_last_state = self.lstm_forw(reversed_feature, last_back_state)
        lstm_back_output = lstm_back_output[0]

        lstm_output = lstm_forw_output + lstm_back_output

        # (B,N/5,C,H,W) --> (BN/5,C,H,W)
        lstm_output = torch.cat(torch.split(lstm_output, split_size_or_sections=1, dim=0), dim=1).squeeze(dim=0)

        ## decoder
        x = self.final_conv(lstm_output)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        out = self.final_sigmoid(x)

        # different outs for train and test
        if B==1 and N==5:
            outs = torch.squeeze(out)
        else:
            outs = torch.stack(torch.split(out, split_size_or_sections=N//5, dim=0), dim=0)

        return outs, lstm_forw_output_last_state, lstm_back_output_last_state


# build the whole network
def build_model():
    return Model(vgg16(base['vgg'], 3))

# weight init
def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)



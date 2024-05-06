from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#numpy và torch là thư viện cho xử lý mảng số và máy học.
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.io import read_image
#torch.nn là module của PyTorch cho mô hình nơ-ron và các lớp liên quan.
import matplotlib.pyplot as plt
import os
import cv2
from tkinter import Listbox, Button, messagebox


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#class gated 2D convolution


#Đây là một lớp định nghĩa kiến trúc của một tầng gated 2D convolution.
#Các tham số như num_in, num_out, ksize, ... là các thông số cần thiết cho việc xây dựng tầng convolution.
# sử dụng deeplearning, mạng GAN để xử lý cấu trúc ảnh

# generative model nghĩa là mô hình có khả năng sinh ra dữ liệu. GAN là mô hình có khả năng sinh ra dữ liệu mới. 
# Ví dụ như những ảnh mặt người ở dưới bạn thấy là do GAN sinh ra, 
# không phải mặt người thật. Dữ liệu sinh ra nhìn như thật nhưng không phải thật.

class GConv(nn.Module):
    def __init__(self,
                 num_in, 
                 num_out, 
                 ksize,  
                 stride = 1,
                 padding ='auto',
                 
                 rate = 1, 
                 activation = nn.ELU(),
                 bias = True):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        num_out_final = num_out if num_out == 3 or activation is None else 2*num_out
        self.activation = activation
        self.ksize = ksize
        padding = rate*(ksize-1)//2 if padding == 'auto' else padding 
        self.padding = padding
        self.stride = stride
        
        self.conv = nn.Conv2d(self.num_in, 
                              num_out_final,
                              kernel_size = ksize,
                              stride = stride,
                              padding = padding,
                              dilation = rate,
                              bias = bias)
    def forward(self,x):
        x = self.conv(x)
        if self.num_out == 3 or self.activation is None:
            return x
        x,y = torch.split(x,self.num_out,dim = 1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x*y
        return x

#Upsampling
#Là lớp định nghĩa kiến trúc của một tầng upsampling sử dụng convolution.

class GDeConv(nn.Module):
    def __init__(self, cnum_in, cnum_out, padding = 1):
        super().__init__()
        self.conv = GConv(cnum_in,cnum_out,ksize=3,stride=1,padding=padding)
    def forward(self,x):
        x=F.interpolate(x,scale_factor=2, mode = 'nearest',
                        recompute_scale_factor= False)
        x = self.conv(x)
        return x


#down sample conv   
#Downsampling path: Trích suất đặc trưng ngữ cảnh và nội dung của ảnh. Downsampling path là kiến trúc thu hẹp kích thước layer
class GDownsamplingBlock(nn.Module):
    def __init__(self, cnum_in, cnum_out, cnum_hidden = None):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden ==None else cnum_hidden
        self.conv1_downsample = GConv(cnum_in, cnum_hidden, 3, 2)
        self.conv2 = GConv(cnum_hidden, cnum_out ,3,1)
        
    def forward(self,x):
        x = self.conv1_downsample(x)
        x = self.conv2(x)
        
        return x

#Upsample Deconv
# Một khối chứa hai tầng convolution được sử dụng để thực hiện quá trình down-sampling.
# Upsampling: Các kỹ thuật giúp tăng kích thước output trong mạng CNN

class GUsamplingBlock(nn.Module):
    def __init__(self, cnum_in, cnum_out, cnum_hidden = None):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden ==None else cnum_hidden
        self.conv1_upsample = GDeConv(cnum_in, cnum_hidden)
        self.conv2 = GConv(cnum_hidden, cnum_out ,3,1)
        
    def forward(self,x):
        x = self.conv1_upsample(x)
        x = self.conv2(x)
        
        return x
  
# chuyển đổi đầu ra của mô hình (tensor) thành ảnh.    
# Tensor hay tiếng Việt gọi là Ten-xơ là đối tượng hình học miêu tả quan hệ tuyến tính giữa các đại lượng vectơ,
# vô hướng, và các tenxơ với nhau.
def output_to_image(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out

#Định nghĩa một mô hình Generator, được xây dựng từ các khối CoarseGenerator và FineGenerator.
#Các phương thức forward và infer được sử dụng để thực hiện quá trình sinh ảnh.
class Generator(nn.Module):
    def __init__(self, cnum_in=5, cnum=48, return_flow=False, checkpoint=None):
        super().__init__()
        self.stage1 = CoarseGenerator(cnum_in, cnum)
        self.stage2 = FineGenerator(cnum, return_flow)
        self.return_flow = return_flow

        if checkpoint is not None:
            generator_state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['G']
            self.load_state_dict(generator_state_dict, strict=True)

        self.eval()

    def forward(self, x, mask):
        xin = x
        # get coarse result
        x_stage1 = self.stage1(x)
        # inpaint input with coarse result
        x = x_stage1*mask + xin[:, 0:3, :, :]*(1.-mask)
        # get refined result
        x_stage2, offset_flow = self.stage2(x, mask)

        if self.return_flow:
            return x_stage1, x_stage2, offset_flow

        return x_stage1, x_stage2

    @torch.inference_mode()
    def infer(self,
              image,
              mask,
              return_vals=['inpainted', 'stage1'],
              device='cuda'):
        """
        Args:
            image: 
            mask:
            return_vals: inpainted, stage1, stage2, flow
        Returns:
        """

        _, h, w = image.shape
        grid = 8

        image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

        image = (image*2 - 1.)  # map image values to [-1, 1] range
        # 1.: masked 0.: unmasked
        mask = (mask > 0.).to(dtype=torch.float32)

        image_masked = image * (1.-mask)  # mask image

        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]  # sketch channel
        x = torch.cat([image_masked, ones_x, ones_x*mask],
                      dim=1)  # concatenate channels

        if self.return_flow:
            x_stage1, x_stage2, offset_flow = self.forward(x, mask)
        else:
            x_stage1, x_stage2 = self.forward(x, mask)

        image_compl = image * (1.-mask) + x_stage2 * mask

        output = []
        for return_val in return_vals:
            if return_val.lower() == 'stage1':
                output.append(output_to_image(x_stage1))
            elif return_val.lower() == 'stage2':
                output.append(output_to_image(x_stage2))
            elif return_val.lower() == 'inpainted':
                output.append(output_to_image(image_compl))
            elif return_val.lower() == 'flow' and self.return_flow:
                output.append(offset_flow)
            else:
                print(f'Invalid return value: {return_val}')

        return output

# ----------------------------------------------------------------------------
#Là một lớp thực hiện Contextual Attention, một phần quan trọng trong mô hình 
# để tập trung vào các vùng quan trọng của hình ảnh.

class ContextualAttention(nn.Module):
    """ Contextual attention layer implementation. \\
        Contextual attention is first introduced in publication: \\
        `Generative Image Inpainting with Contextual Attention`, Yu et al \\
        Args:
            ksize: Kernel size for contextual attention
            stride: Stride for extracting patches from b
            rate: Dilation for matching
            softmax_scale: Scaled softmax for attention
    """

    def __init__(self,
                 ksize=3,
                 stride=1,
                 rate=1,
                 fuse_k=3,
                 softmax_scale=10.,
                 n_down=2,
                 fuse=False,
                 return_flow=False,
                 device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.device_ids = device_ids
        self.n_down = n_down
        self.return_flow = return_flow
        self.register_buffer('fuse_weight', torch.eye(
            fuse_k).view(1, 1, fuse_k, fuse_k))

    def forward(self, f, b, mask=None):
        """
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
        """
        device = f.device
        # get shapes
        raw_int_fs, raw_int_bs = list(f.size()), list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksize=kernel,
                                      stride=self.rate*self.stride,
                                      rate=1, padding='auto')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate,
                          mode='nearest', recompute_scale_factor=False)
        b = F.interpolate(b, scale_factor=1./self.rate,
                          mode='nearest', recompute_scale_factor=False)
        int_fs, int_bs = list(f.size()), list(b.size())   # b*c*h*w
        # split tensors along the batch dimension
        f_groups = torch.split(f, 1, dim=0)
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksize=self.ksize,
                                  stride=self.stride,
                                  rate=1, padding='auto')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros(
                [int_bs[0], 1, int_bs[2], int_bs[3]], device=device)
        else:
            mask = F.interpolate(
                mask, scale_factor=1./((2**self.n_down)*self.rate), mode='nearest', recompute_scale_factor=False)
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksize=self.ksize,
                                  stride=self.stride,
                                  rate=1, padding='auto')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]

        mm = (torch.mean(m, dim=[1, 2, 3], keepdim=True) == 0.).to(
            torch.float32)
        mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(torch.sum(torch.square(wi), dim=[
                                1, 2, 3], keepdim=True)).clamp_min(1e-4)
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            yi = F.conv2d(xi, wi_normed, stride=1, padding=(
                self.ksize-1)//2)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                # (B=1, I=1, H=32*32, W=32*32)
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                # (B=1, C=1, H=32*32, W=32*32)
                yi = F.conv2d(yi, self.fuse_weight, stride=1,
                              padding=(self.fuse_k-1)//2)
                # (B=1, 32, 32, 32, 32)
                yi = yi.contiguous().view(
                    1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])
                yi = yi.permute(0, 2, 1, 4, 3)

                yi = yi.contiguous().view(
                    1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = F.conv2d(yi, self.fuse_weight, stride=1,
                              padding=(self.fuse_k-1)//2)
                yi = yi.contiguous().view(
                    1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()

            # (B=1, C=32*32, H=32, W=32)
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            if self.return_flow:
                offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

                if int_bs != int_fs:
                    # Normalize the offset value to match foreground dimension
                    times = (int_fs[2]*int_fs[3])/(int_bs[2]*int_bs[3])
                    offset = ((offset + 1).float() * times - 1).to(torch.int64)
                offset = torch.cat([torch.div(offset, int_fs[3], rounding_mode='trunc'),
                                    offset % int_fs[3]], dim=1)  # 1*2*H*W
                offsets.append(offset)

            # deconv for patch pasting
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(
                yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y = y.contiguous().view(raw_int_fs)

        if not self.return_flow:
            return y, None

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2], device=device).view(
            [1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3], device=device).view(
            [1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        offsets = offsets - torch.cat([h_add, w_add], dim=1)
        # to flow image
        flow = torch.from_numpy(flow_to_image(
            offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate,
                                 mode='bilinear', align_corners=True)

        return y, flow

# ----------------------------------------------------------------------------
#Dùng để chuyển đổi thông tin về optical flow

# trong deep learning Optical Flow là sự chuyển động của đối tượng giữa 
# 2 frame (khung hình) liên tiếp trong 1 đoạn ảnh, video
def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))

# ----------------------------------------------------------------------------
# tiếp theo hàm flow_to_image, hàm này để chuyển ảnh thành ảnh màu RGB
def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img

# ----------------------------------------------------------------------------

def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC,
               2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM,
               0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel

# ----------------------------------------------------------------------------

#Hàm extract_image_patches có chức năng trích xuất các khối hình ảnh trượt qua hình ảnh đầu vào.

def extract_image_patches(images, ksize, stride, rate, padding='auto'):
    """
    Extracts sliding local blocks \\
    see also: https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    """

    padding = rate*(ksize-1)//2 if padding == 'auto' else padding

    unfold = torch.nn.Unfold(kernel_size=ksize,
                             dilation=rate,
                             padding=padding,
                             stride=stride)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

# ----------------------------------------------------------------------------

#Lớp Conv2DSpectralNorm là một lớp convolution được mở rộng từ nn.Conv2d trong PyTorch,
# được thiết kế để áp dụng Spectral Normalization trước mỗi lần gọi.

#Spectral Normalization là một kỹ thuật chuẩn hóa được thiết kế để ổn định và 
# cải thiện quá trình đào tạo của mô hình máy học

class Conv2DSpectralNorm(nn.Conv2d):
    """Convolution layer that applies Spectral Normalization before every call."""

    def __init__(self, cnum_in,
                 cnum_out, kernel_size, stride, padding=0, n_iter=1, eps=1e-12, bias=True):
        super().__init__(cnum_in,
                         cnum_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        self.register_buffer("weight_u", torch.empty(self.weight.size(0), 1))
        nn.init.trunc_normal_(self.weight_u)
        self.n_iter = n_iter
        self.eps = eps

    def l2_norm(self, x):
        return F.normalize(x, p=2, dim=0, eps=self.eps)

    def forward(self, x):

        weight_orig = self.weight.flatten(1).detach()

        for _ in range(self.n_iter):
            v = self.l2_norm(weight_orig.t() @ self.weight_u)
            self.weight_u = self.l2_norm(weight_orig @ v)

        sigma = self.weight_u.t() @ weight_orig @ v
        self.weight.data.div_(sigma)

        x = super().forward(x)

        return x

# ----------------------------------------------------------------------------
#Hàm DConv (Discriminator Convolution) là một phần của mô hình
# trong một mô hình GAN (Generative Adversarial Network) 

#thường sử dụng các lớp convolution để học các đặc trưng của ảnh và
# đưa ra dự đoán về tính thật hay giả của ảnh

class DConv(nn.Module):
    def __init__(self, cnum_in,
                 cnum_out, ksize=5, stride=2, padding='auto'):
        super().__init__()
        padding = (ksize-1)//2 if padding == 'auto' else padding
        self.conv_sn = Conv2DSpectralNorm(
            cnum_in, cnum_out, ksize, stride, padding)
        #self.conv_sn = spectral_norm(nn.Conv2d(cnum_in, cnum_out, ksize, stride, padding))
        self.leaky = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv_sn(x)
        x = self.leaky(x)
        return x

# ----------------------------------------------------------------------------
# trong mô hình GAN Hàm Discriminator chịu trách nhiệm học cách phân biệt giữa các ảnh thật 
# và ảnh được tạo ra bởi mô hình generator.

class Discriminator(nn.Module):
    def __init__(self, cnum_in, cnum):
        super().__init__()
        self.conv1 = DConv(cnum_in, cnum)
        self.conv2 = DConv(cnum, 2*cnum)
        self.conv3 = DConv(2*cnum, 4*cnum)
        self.conv4 = DConv(4*cnum, 4*cnum)
        self.conv5 = DConv(4*cnum, 4*cnum)
        self.conv6 = DConv(4*cnum, 4*cnum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = nn.Flatten()(x)

        return x

#coarse #fine
# Nhiệm vụ chính của generator là tạo ra ảnh có chất lượng cao từ một điểm dữ liệu ngẫu nhiên.
class CoarseGenerator(nn.Module):
    def __init__(self, cnum_in, cnum):
        super().__init__()
        self.conv1 = GConv(cnum_in, cnum //2,5,1, padding= 2)
        
        #downsample conv
        self.down_block1 = GDownsamplingBlock(cnum//2,cnum)
        self.down_block2 = GDownsamplingBlock(cnum,2*cnum)

        #Bottleneck
        self.conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1)
        self.conv_bn2 = GConv(2*cnum, 2*cnum, 3, rate=2, padding= 2)
        self.conv_bn3 = GConv(2*cnum, 2*cnum, 3, rate=4, padding= 4)
        self.conv_bn4 = GConv(2*cnum, 2*cnum, 3, rate=8, padding= 8)
        self.conv_bn5 = GConv(2*cnum, 2*cnum, 3, rate=16, padding= 16)
        self.conv_bn6 = GConv(2*cnum, 2*cnum, 3, 1)
        self.conv_bn7 = GConv(2*cnum, 2*cnum, 3, 1)

        #upsample deconv
        self.up_block1 = GUsamplingBlock(2*cnum, cnum)
        self.up_block2 = GUsamplingBlock(cnum,cnum//4, cnum_hidden= cnum//2)
        
        #RGB
        self.conv_to_rgb = GConv(cnum//4,3,3,1, activation= None)
        self.tanh = nn.Tanh()

    def forward(self,x):
        x = self.conv1(x)
        #down
        x = self.down_block1(x)
        x = self.down_block2(x)
        #Bottleneck
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        x = self.conv_bn3(x)
        x = self.conv_bn4(x)
        x = self.conv_bn5(x)
        x = self.conv_bn6(x)
        x = self.conv_bn7(x)
        #up
        x = self.up_block1(x)
        x = self.up_block2(x)
        
        #RGB
        x = self.conv_to_rgb(x)
        x = self.tanh(x)
        
        return x

class FineGenerator(nn.Module):
    def __init__(self, cnum, return_flow = False):
        super().__init__()
        
        self.conv_conv1 = GConv(3, cnum //2,5,1,padding=2)
        
        #down
        self.conv_down_block1 = GDownsamplingBlock(cnum//2,cnum,cnum_hidden=cnum//2)
        self.conv_down_block2 = GDownsamplingBlock(cnum,2*cnum,cnum_hidden=cnum)
        
        #Bottleneck
        self.conv_conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1)
        self.conv_conv_bn2 = GConv(2*cnum, 2*cnum, 3, rate=2, padding= 2)
        self.conv_conv_bn3 = GConv(2*cnum, 2*cnum, 3, rate=4, padding= 4)
        self.conv_conv_bn4 = GConv(2*cnum, 2*cnum, 3, rate=8, padding= 8)
        self.conv_conv_bn5 = GConv(2*cnum, 2*cnum, 3, rate=16, padding= 16)
        
        #contextual Attention
        self.ca_conv1 = GConv(3, cnum//2,5,1,2)
        self.ca_down_block1 = GDownsamplingBlock(cnum//2,cnum,cnum_hidden=cnum//2)
        self.ca_down_block2 = GDownsamplingBlock(cnum, 2*cnum)
        
        #bottlenck
        self.ca_conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1, activation= nn.ReLU())
        #bodyca
        self.contextual_attention = ContextualAttention(ksize= 3, stride=1,rate=2,
                                                        fuse_k = 3, softmax_scale=10, 
                                                        fuse=True, device_ids= None , 
                                                        return_flow=return_flow, n_down = 2)
        self.ca_conv_bn4 = GConv(2*cnum, 2*cnum,3,1)
        self.ca_conv_bn5 = GConv(2*cnum, 2*cnum,3,1)
        self.conv_bn6 = GConv(4*cnum, 2*cnum,3,1)
        self.conv_bn7 = GConv(2*cnum, 2*cnum,3,1)
        
        #upsample
        self.up_block1 = GUsamplingBlock(2*cnum, cnum)
        self.up_block2 = GUsamplingBlock(cnum,cnum//4,cnum_hidden=cnum//2)
        #RGB
        self.conv_to_rgb = GConv(cnum//4,3,3,1, activation= None)
        self.tanh = nn.Tanh()        
    
    def forward(self,x,mask):
        xnow = x
        x = self.conv_conv1(xnow)
        
        x = self.conv_down_block1(x)
        x = self.conv_down_block2(x)
        
        x = self.conv_conv_bn1(x)
        x = self.conv_conv_bn2(x)
        x = self.conv_conv_bn3(x)
        x = self.conv_conv_bn4(x)
        x = self.conv_conv_bn5(x)
        x_hallu = x
        
        x = self.ca_conv1(xnow)
        
        #down
        x = self.ca_down_block1(x)
        x = self.ca_down_block2(x)
        
        #bottleneck
        x = self.ca_conv_bn1(x)
        x, offset_flow = self.contextual_attention(x,x,mask)
        x = self.ca_conv_bn4(x)
        x = self.ca_conv_bn5(x)
        pm = x
        
        #concatenate
        
        x = torch.cat([x_hallu,pm],dim=1)
        x = self.conv_bn6(x)
        x = self.conv_bn7(x)
        
        #up sample
        
        x = self.up_block1(x)
        x = self.up_block2(x)
        
        x = self.conv_to_rgb(x)
        x = self.tanh(x)
        
        return x ,offset_flow
    
#tạo giao diện cửa sổ đầu tiên 
    
class MachineVisionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("MACHINE VISION")
        self.master.geometry("1000x750")

        # Tạo logo HCMUTE
        image1 = Image.open("logoHCMUTE.png")
        self.photo = ImageTk.PhotoImage(image1)
        self.image_label = Label(self.master, image=self.photo)
        self.image_label.place(x=10, y=10)

        # Các thông tin khác
        self.create_labels()
        self.create_button()
        

    def create_labels(self):
        label1 = Label(self.master, text='KHOA ĐÀO TẠO QUỐC TẾ', font=("Arial", 20), fg='blue')
        label1.place(x=640, y=30)

        label2 = Label(self.master, text='CÔNG NGHỆ KỸ THUẬT CƠ ĐIỆN TỬ', font=("Arial", 16), fg='blue')
        label2.place(x=625, y=75)

        label3 = Label(self.master, text='ỨNG DỤNG XỬ LÝ ẢNH CHO', font=("Arial", 26), fg='RED')
        label3.place(x=280, y=160)

        label4 = Label(self.master, text='CỬA HÀNG TRANG TRÍ NỘI THẤT', font=("Arial", 26), fg='RED')
        label4.place(x=235, y=200)

        MOTA = Label(self.master, text='Mô tả: phát triển một ứng dụng dành cho các cửa hàng nội thất, hỗ trợ quản lý, thêm,',
                     font=("Arial", 15), fg='black')
        MOTA.place(x=145, y=450)
        MOTA2 = Label(self.master,
                      text=' xóa vật thể để khách hàng có thể sắp xếp nội thất phù hợp với sở thích cá nhân.',
                      font=("Arial", 15), fg='black')
        MOTA2.place(x=188, y=490)

        label5 = Label(self.master, text='GVHD: TS. Nguyễn Văn Thái', font=("Arial", 20), fg='black')
        label5.place(x=490, y=550)

        label6 = Label(self.master, text='SVTH:', font=("Arial", 20), fg='black')
        label6.place(x=490, y=590)

        label7 = Label(self.master, text=' Phạm Anh Kiệt             20146220', font=("Arial", 20), fg='black')
        label7.place(x=572, y=590)

        label8 = Label(self.master, text=' Nguyễn Kim Long         20146222', font=("Arial", 20), fg='black')
        label8.place(x=570, y=630)

        label9 = Label(self.master, text=' Trương Lê Duy Nhân    20146263', font=("Arial", 20), fg='black')
        label9.place(x=570, y=670)

    def create_button(self):
        image_path = "start.png"
        original_image = Image.open(image_path)
        resized_image = original_image.resize((140, 120))
        self.start_image = ImageTk.PhotoImage(resized_image)
        self.start_button = tk.Button(self.master, image=self.start_image, command=self.switch_to_class2)
        self.start_button.place(x=420, y=265)
        label10 = Label(self.master, text='BẮT ĐẦU CHƯƠNG TRÌNH', font=("Arial", 20), fg='black')
        label10.place(x=320, y=400)

    def switch_to_class2(self):
        self.master.destroy()  # Đóng cửa sổ của Class 1
        root = tk.Tk()
        my_gui = Remove_Object(root, rcnn_segmentation, rcnn_transforms, deepfill, path1, path2)
        root.mainloop()
        
        
#chương trình chính để thêm, xóa vật thể

class Remove_Object():
    def __init__(self, master, segmentModel, rcnn_transforms, deepfill, img_path, img_path2):
        self.master = master
        self.root = self.master
        self.segmentModel = segmentModel
        self.rcnn_transforms = rcnn_transforms
        self.img_path = img_path
        self.img_path2 = img_path2
        self.deepfill = deepfill
        self.img_orig = None  # Initialize img_orig
        self.processed_once = False
        
        self.root.title("CÁC PHÍM CHỨC NĂNG")
        
        #thêm vật
        self.objects_to_add = ['anhTVnhinthang1.png','anhtulanh2ngannhinthang1.png','zyro-image (1).png','anhtulanh2ngannhinsangtrai1.png','anhtulanh2ngansangphai.png']
        self.background_image = cv2.imread(self.img_path2, cv2.IMREAD_COLOR)
        self.objects_state = []
        self.selected_object_index = None
        self.start_x, self.start_y = -1, -1
        self.is_dragging = False
        self.result_window = None  # Keep track of the Result window

        self.create_widgets()

        # Add Tkinter widgets
        imageIOT = Image.open("logoIOT.png")
        self.photo1 = ImageTk.PhotoImage(imageIOT)       
        self.labelIOT = Label(self.master, image=self.photo1)
        self.labelIOT.pack()
        
        self.labelten = Label(self.master, text='CỬA HÀNG NỘI THẤT', font=("Arial", 25), fg='blue')
        self.labelten.pack()
        
        
        
        
        self.button_run = tk.Button(self.master, text="XÓA NỘI THẤT", command=self.run, width=31, height=2, bg="orange", fg="black", font = ("Arial", 20))
        self.button_run.pack()
        # Tạo nút mở cửa sổ danh sách vật thể
        
        self.open_result_button = Button(self.master, text='THÊM NỘI THẤT', command=self.open_result_window, width=31, height=2, bg="orange", fg="black", font = ("Arial", 20))
        self.open_result_button.pack()
        
        open_list_button = Button(self.root, text='MỞ DANH SÁCH NỘI THẤT',command=self.open_object_list_window, width=31, height=2, bg="orange", fg="black", font = ("Arial", 20))
        open_list_button.pack()

        # Tạo nút xóa đối tượng
        delete_button = Button(self.root, text='XÓA NỘI THẤT THÊM TỪ DANH SÁCH',command=self.delete_object, width=31, height=2, bg="orange", fg="black", font = ("Arial", 20))
        delete_button.pack()
        
        
        
        
    def img_transform(self):
        if self.processed_once == False:
            
            img = read_image(self.img_path)
            self.processed_once = True
        else:
            img = read_image(self.img_path2)
            
        _,h,w = img.shape
        size = min(h,w)
        if size > 512:
            img.T.resize(512, max_size = 680, antialias = True)(img)
            
        image_transformed = self.rcnn_transforms(img)
        return image_transformed
    def selectObject(self):
        img = self.img_orig.detach().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        points = []
        draw = False
        temp = img.copy()
        def click(event, x, y, flags, params):
            nonlocal points,draw,img,temp
            if event == cv2.EVENT_LBUTTONDOWN:
                draw = True
                points = [x,y]
            elif event ==cv2.EVENT_MOUSEMOVE:
                if draw:
                    img = temp.copy()
                    cv2.rectangle(img,(points[0],points[1]),(x,y),(0,0,255),2)
                    cv2.imshow('select',img)
                    
            elif event ==cv2.EVENT_LBUTTONUP:
                draw = False
                points +=[x,y]
        
        cv2.namedWindow('select')
        cv2.setMouseCallback('select',click)
        while True:
            cv2.imshow('select',img)
            key = cv2.waitKey(1)
            if key == ord('s'):
                break
        cv2.destroyAllWindows()
        return points
# IoU là độ đo đánh giá các mô hình nhận diện thực thể. Phép đo này có thể đánh giá các mô hình khác nhau như RCNN, 
# Fast-RCNN, Faster- RCNN hay YOLO. Sau khi huấn luyện xong một mô hình nhận diện thực thể, 
# chúng ta cần đánh giá xem mô hình hoạt động chính xác đến mức độ nào.    
    def iou(self,boxes_rcnn, boxes_rect):
        x1 = np.array ([boxes_rcnn[:,0],boxes_rect[:,0]]).max(axis=0)
        y1 = np.array ([boxes_rcnn[:,1],boxes_rect[:,1]]).max(axis=0)
        
        x2 = np.array ([boxes_rcnn[:,2],boxes_rect[:,2]]).min(axis=0)
        y2 = np.array ([boxes_rcnn[:,3],boxes_rect[:,3]]).min(axis=0)
    
        w = x2 - x1
        h = y2 - y1
        
        #condition
        w[w<0] = 0
        h[h<0] = 0
        intersection = w*h
        area_a = (boxes_rcnn[:,2]- boxes_rcnn[:,0]) * (boxes_rcnn[:,3] - boxes_rcnn[:,1])
        area_b = (boxes_rect[:,2]- boxes_rect[:,0]) * (boxes_rect[:,3] - boxes_rect[:,1])
        
        
        union = area_a + area_b - intersection
        
        return intersection/(union + 0.00001)
        
    def find_mask(self, rcnn_output, rectangle):
        bounding_boxes = rcnn_output['boxes'].detach().numpy()
        masks = rcnn_output['masks']
        rect_boxes = np.array([rectangle], dtype= object)
        rect_boxes = np.repeat(rect_boxes, bounding_boxes.shape[0], axis=0)
        
        iou_total = self.iou(bounding_boxes, rect_boxes)
        mask_index = np.argmax(iou_total)
        
        return masks[mask_index]
    
    def remove_fill(self, image, mask):

        res = self.deepfill.infer(image, mask, return_vals = ['inpainted'])
        return res[0]
    
                
    def run(self):
        cv2.destroyAllWindows()
        print("load image")
        img  = self.img_transform()
        self.img = img
        self.img_orig = img.permute(1,2,0)
        points = self.selectObject()
        print("segmentation process")
        data =  rcnn_segmentation([img])
        out = data[0]
        self.highest_mask = self.find_mask(out, points)
        self.res = self.remove_fill(self.img, self.highest_mask )
        
        cv2.imwrite("anhketqua.jpg", cv2.cvtColor(self.res, cv2.COLOR_RGB2BGR))
        
        self.display()
        
    # thêm vật thể    
    def create_widgets(self):

        cv2.namedWindow('Result')
        cv2.setMouseCallback('Result', self.mouse_callback)

        self.update_background_image()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, obj_state in enumerate(self.objects_state):
                obj_x, obj_y = obj_state['position']
                if obj_x <= x <= obj_x + obj_state['size'][1] and obj_y <= y <= obj_y + obj_state['size'][0]:
                    self.selected_object_index = i
                    self.start_x, self.start_y = x - obj_x, y - obj_y
                    self.is_dragging = True
                    break

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            if self.selected_object_index is not None:
                obj_state = self.objects_state[self.selected_object_index]
                obj_height, obj_width = obj_state['size']

                new_x, new_y = x - self.start_x, y - self.start_y
                new_x = max(0, min(self.background_image.shape[1] - obj_width, new_x))
                new_y = max(0, min(self.background_image.shape[0] - obj_height, new_y))
                obj_state['position'] = (new_x, new_y)

                self.update_background_image()

    def open_object_list_window(self):
        object_list_window = tk.Toplevel(self.master)
        object_list_window.title('Object List')

        object_listbox = Listbox(object_list_window)
        for obj in self.objects_to_add:
            object_listbox.insert(tk.END, obj)

        def select_object():
            selected_object_path = self.objects_to_add[object_listbox.curselection()[0]]
            selected_object = cv2.imread(selected_object_path, cv2.IMREAD_UNCHANGED)

            obj_height, obj_width = selected_object.shape[:2]
            selected_object_resized = cv2.resize(selected_object, (obj_width, obj_height))

            self.objects_state.append({'image_resized': selected_object_resized, 'position': (0, 0),
                                       'size': (obj_height, obj_width)})

            self.update_background_image()
            object_list_window.destroy()

        select_button = Button(object_list_window, text='Select Object', command=select_object)
        select_button.pack()

        object_listbox.pack()

    def delete_object(self):
        if self.selected_object_index is not None:
            del self.objects_state[self.selected_object_index]
            self.selected_object_index = None
            self.update_background_image()
        else:
            messagebox.showinfo('Info', 'No object selected.')

    def update_background_image(self):
        self.background_image = cv2.imread(self.img_path2, cv2.IMREAD_COLOR).copy()

        for obj_state in self.objects_state:
            obj_x, obj_y = obj_state['position']
            selected_object_resized = obj_state['image_resized']
            alpha_mask = selected_object_resized[:, :, 3] / 255.0

            min_row = max(0, obj_y)
            max_row = min(self.background_image.shape[0], obj_y + obj_state['size'][0])
            min_col = max(0, obj_x)
            max_col = min(self.background_image.shape[1], obj_x + obj_state['size'][1])

            M = np.array([
                [1, 0, obj_x - min_col],
                [0, 1, obj_y - min_row]
            ], dtype=np.float32)

            selected_object_resized = cv2.warpAffine(selected_object_resized, M, (max_col - min_col, max_row - min_row))

            for c in range(0, 3):
                self.background_image[min_row:max_row, min_col:max_col, c] = \
                    self.background_image[min_row:max_row, min_col:max_col, c] * (
                                1 - alpha_mask[:max_row - min_row, :max_col - min_col]) + \
                    selected_object_resized[:max_row - min_row, :max_col - min_col, c] * \
                    alpha_mask[:max_row - min_row, :max_col - min_col]

        cv2.imshow('Result', self.background_image)

    def open_result_window(self):
        if self.result_window is not None and not self.result_window.winfo_exists():
            cv2.destroyAllWindows()
            # If the Result window is closed, set it to None
            self.result_window = None

        if self.result_window is None:
            
            # cv2.destroyAllWindows()
            # self.result_window = tk.Toplevel(self.master)
            # self.result_window.title('Result')

            # cv2.namedWindow('Result')
            # cv2.setMouseCallback('Result', self.mouse_callback)

            # # Update the background image when reopening the Result window
            # self.update_background_image()
            
            cv2.destroyAllWindows()
            self.update_background_image()
            
            cv2.namedWindow('Result')
            cv2.setMouseCallback('Result', self.mouse_callback)

    
    def display(self):
        
        # Assuming self.img_orig is a PIL Image
        hinhgoc = np.array(self.img_orig)
        ketqua = np.array(self.res)
        
        # Convert from RGB to BGR if needed
        img_bgr = cv2.cvtColor(hinhgoc, cv2.COLOR_RGB2BGR)
        img_bgr1 = cv2.cvtColor(ketqua, cv2.COLOR_RGB2BGR)
        
        # Display using OpenCV
        cv2.imshow('anh goc', img_bgr)
        cv2.imshow('anh ket qua', img_bgr1)



if __name__ == "__main__":
    deepfill_path = r'file.pth'

    # Assuming Generator, read_image, and MaskRCNN_ResNet50_FPN_Weights are properly defined
    deepfill = Generator(checkpoint=deepfill_path, return_flow=True)

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    rcnn_transforms = weights.transforms()
    rcnn_segmentation = maskrcnn_resnet50_fpn(weights=weights, progress=False).eval()
    path1 = r'hinhtest.jpg'
    path2 = r'anhketqua.jpg'
    
    root1 = tk.Tk()
    app = MachineVisionApp(root1)
    root1.mainloop()

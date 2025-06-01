# -*- coding: utf-8 -*-
"""

Self-Net Inference

This code is adapted from the code provided with the Self-Net research article 
(published in Light: Science & Applications,  https://doi.org/10.1038/s41377-023-01230-2) 
titled "Deep self-learning enables fast, high-fidelity isotropic resolution restoration 
for volumetric fluorescence microscopy" 
available here:https://zenodo.org/records/7882519

This file contains an improved pytorch version for tile gneration, replacing the
matlab script and need for any proprietary software
It also comibes all modules and functions into one script for the purpose of being called
as a single script with arguments from RESPAN

Luke Hammond
luke.hammond@osumc.edu
Director of Quantitative Imaging
Department of Neurology
The Ohio State University

"""

import os
import torch
from skimage.transform import resize
import numpy as np
from tifffile import imread, imwrite
import threading
import argparse
import sys
import locale

import torch.nn as nn
from torch.nn import init
import functools
import math
import torch.nn.functional as F

# Force UTF-8 encoding for frozen applications
if getattr(sys, 'frozen', False):
    # Set environment variables for UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Reconfigure stdout/stderr
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def reslice_old_selfnet(img,position,x_res,z_res):
    scale=z_res/x_res
    z,y,x=img.shape
    if position=='xz':
        reslice_img=np.transpose(img,[1,0,2])

        scale_img=np.zeros((y,round(z*scale),x),dtype=np.uint16)
        for i in range(y):
            scale_img[i]=cv2.resize(reslice_img[i],(x,round(z*scale)),interpolation=cv2.INTER_CUBIC)

    else:
        reslice_img=np.transpose(img,[2,0,1])
        scale_img = np.zeros((x, round(z * scale), y), dtype=np.uint16)
        for i in range(x):
            scale_img[i] = cv2.resize(reslice_img[i], (y,round(z * scale)), interpolation=cv2.INTER_CUBIC)

    return scale_img

def reslice(img, position, x_res, z_res):
    scale = z_res / x_res
    z, y, x = img.shape
    
    if position == 'xz':
        # Reslice along the XZ plane
        resliced_img = np.transpose(img, (1, 0, 2))  # y, z, x -> z, y, x
        # Scale image along the z-axis
        scaled_img = np.zeros((y, round(z * scale), x), dtype=np.uint16)
        for i in range(y):
            scaled_img[i] = resize(resliced_img[i], (round(z * scale), x), 
                                   order=3, mode='reflect', preserve_range=True).astype(np.uint16)
    
    elif position == 'yz':
        # Reslice along the YZ plane
        resliced_img = np.transpose(img, (2, 0, 1))  # y, z, x -> x, z, y
        # Scale image along the z-axis
        scaled_img = np.zeros((x, round(z * scale), y), dtype=np.uint16)
        for i in range(x):
            scaled_img[i] = resize(resliced_img[i], (round(z * scale), y), 
                                   order=3, mode='reflect', preserve_range=True).astype(np.uint16)
    
    return scaled_img


def output_img(deblur_net,device,min_v,max_v,write_stack,raw_img):
    deblur_net.eval()
    device_cpu = torch.device('cpu')

    batch_size=3

    z_shape=raw_img.shape[0]

    idx = z_shape// batch_size

    res=z_shape-idx*batch_size

    input_img = (raw_img.astype(np.float32) - min_v) / (max_v - min_v)
    input_img[input_img> 1] = 1
    input_img[input_img < 0] = 0
    input_img = np.expand_dims(input_img, axis=1)

    input_tensor = torch.from_numpy(input_img)

    for ii in range(idx):
        with torch.no_grad():
            test_tensor=input_tensor[ii * batch_size:(ii + 1) * batch_size].to(device)
            net_output = deblur_net(test_tensor)
            print('{}/{}'.format((ii + 1) * batch_size, z_shape))
        net_output = net_output.squeeze_(1).to(device_cpu).numpy()
        net_output = net_output * (max_v - min_v) + min_v
        net_output = np.clip(net_output, 0, max_v).astype(np.uint16)

        write_stack[ii * batch_size:(ii + 1) * batch_size] = net_output


    if res!=0:
        test_tensor = input_tensor[idx * batch_size:].to(device)
        with torch.no_grad():
            net_output = deblur_net(test_tensor)
            print('{}/{}'.format(z_shape, z_shape))

        net_output = net_output.squeeze_(1).to(device_cpu).numpy()
        net_output = net_output * (max_v - min_v) + min_v
        net_output = np.clip(net_output, 0, max_v).astype(np.uint16)

        write_stack[idx * batch_size:] = net_output

'''
Self-Net code is partially built on Cycle-consistent Generative Adversarial Networks (**CycleGANs**) 
[[paper]](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html), 
which makes unsupervised training of CNNs possible and is really illuminating.
[[code]]https://github.com/junyanz/CycleGAN
'''


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type=='square':
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net,device,init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, device, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02):

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'deblur_net':
        net = Deblur_Net(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)

    elif netG=='care':
        net=Care_net(input_nc, output_nc, ngf, norm_layer=norm_layer)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, device,init_type, init_gain)


def define_D(input_nc, ndf, netD, device,n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02):
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, device,init_type, init_gain)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()

        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """

        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)

        return loss



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)



class Deblur_Net(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        assert(n_blocks >= 0)
        super(Deblur_Net, self).__init__()
        use_bias = True

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),nn.LeakyReLU(0.2,True)]

        n_1 = 2
        for i in range(n_1):  # add layers
            model += [nn.Conv2d(ngf , ngf , kernel_size=3, stride=1, padding=1, bias=use_bias),nn.LeakyReLU(0.2,True)]

        for i in range(n_blocks):       # add ResNet blocks

            model += [Res_Block(ngf, padding_type=padding_type, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_1):  # add layers
            model += [nn.Conv2d(ngf ,ngf,kernel_size=3, stride=1,padding=1,bias=use_bias),nn.LeakyReLU(0.2,True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class Res_Block(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(Res_Block, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.LeakyReLU(0.2,True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out



class Care_net(nn.Module):
    def __init__(self, input_nc=2, output_nc=2, ngf=32,norm_layer=nn.BatchNorm2d):
        super(Care_net,self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downrelu=nn.LeakyReLU(0.2, True)
        uprelu=nn.ReLU(True)

        conv1 = [nn.Conv2d(input_nc,ngf,kernel_size=3,stride=1,padding=1,bias=use_bias), downrelu,
                 nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), downrelu]
        self.down1=nn.Sequential(*conv1)

        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        conv2=[nn.Conv2d(ngf,ngf*2,kernel_size=3,stride=1,padding=1,bias=use_bias),norm_layer(ngf), downrelu,
                 nn.Conv2d(ngf*2,ngf*2, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), downrelu]

        self.down2 = nn.Sequential(*conv2)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        conv3 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf),
                 downrelu,
                 nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf),
                 downrelu]

        self.middle=nn.Sequential(*conv3)

        self.upsample1=nn.UpsamplingBilinear2d(scale_factor=2)
        up1=[nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf),
                 uprelu,
                 nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf),
             uprelu]

        self.up1=nn.Sequential(*up1)

        self.upsample2=nn.UpsamplingBilinear2d(scale_factor=2)
        up2=[nn.Conv2d(ngf * 2, ngf * 1, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf),
             uprelu,
                 nn.Conv2d(ngf , ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf),
             uprelu]

        self.up2=nn.Sequential(*up2)

        self.final_conv=nn.Conv2d(ngf * 1, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)

    def forward(self, x):
        conv1=self.down1(x)               #32*128*128
        down1=self.maxpool1(conv1)        #32*64*64

        conv2=self.down2(down1)           #64*64*64
        down2=self.maxpool2(conv2)        #64*32*32

        middle=self.middle(down2)         #64*32*32

        up1=self.upsample1(middle)        #64*64*64
        up1=torch.cat((conv2,up1),1)      #128*64*64
        up1=self.up1(up1)                 #32*64*64

        up2=self.upsample2(up1)           #32*128*128
        up2=torch.cat((conv1,up2),1)      #64*128*128
        up2=self.up2(up2)                 #32*128*128

        final=self.final_conv(up2)        #2*128*128


        return final



class myThread (threading.Thread):
    def __init__(self, threadID, name,deblur_net,device,min_v,max_v,write_stack,raw_img):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.deblur_net=deblur_net
        self.device=device
        self.min_v=min_v
        self.max_v=max_v
        self.raw_img=raw_img
        self.write_stack=write_stack

    def run(self):
        print ("start threading" + self.name)
        output_img(self.deblur_net, self.device, self.min_v,self.max_v,self.write_stack,self.raw_img)
        print ("quit threading" + self.name)


def upsample_block(raw_img,x_res,z_res,deblur_netA,deblur_netB, min_v, max_v):

    device1=torch.device('cuda:0')
    device2=torch.device('cuda:1')


    xz_img=reslice(raw_img,'xz',x_res,z_res)
    yz_img=reslice(raw_img,'yz',x_res,z_res)

    print(xz_img.shape,yz_img.shape)


    out_xz_img=np.zeros_like(xz_img,dtype=np.uint16)
    out_yz_img=np.zeros_like(yz_img,dtype=np.uint16)

    thread1=myThread(1,'thread:1',deblur_netA,device1,min_v,max_v,out_xz_img,xz_img)
    thread2=myThread(2,'thread:2',deblur_netB,device2,min_v,max_v,out_yz_img,yz_img)

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    re_out_xz=np.transpose(out_xz_img,[1,0,2])
    re_out_yz=np.transpose(out_yz_img,[1,2,0])

    fusion_stack=re_out_xz/2+re_out_yz/2
    fusion_stack=np.array(fusion_stack,dtype=np.uint16)


    return fusion_stack

   

#############################################################
#Input variables


# Create the parser
parser = argparse.ArgumentParser(description='.')

# Add the arguments
parser.add_argument('--input_dir', type=str, required=True, help='The input directory')
parser.add_argument('--neuron_ch', type=int, default=0, help='the channel for inference')
parser.add_argument('--model_path', type=str, required=True, help='The model path')
parser.add_argument('--min_v', type=int, default=0, help='The minimum intensity')
parser.add_argument('--max_v', type=int, default=65535, help='The maximum intensity')
parser.add_argument('--scale', type=float, default=0.21, help='The resolution scaling factor')
parser.add_argument('--z_step', type=float, default=1, help='The final z-resolution')

# Parse the arguments
args = parser.parse_args()

#  use the arguments
input_dir = args.input_dir
model_path = args.model_path
min_v = args.min_v
max_v = args.max_v
scale = args.scale
z_step = args.z_step

########
#self net code says scale and Z step... but its actually x_res and z_res.. adjust accordingly!
output_dir = os.path.join(input_dir, 'selfnet')
os.makedirs(output_dir, exist_ok=True)

device1 = torch.device('cuda:0')
device2 = torch.device('cuda:1')

deblur_net_A = define_G(input_nc=1, output_nc=1, ngf=64, netG='deblur_net', device=device1,use_dropout=False,norm='instance')
deblur_net_A.load_state_dict(torch.load(model_path,map_location={'cuda:1':'cuda:0'}))

deblur_net_B = define_G(input_nc=1, output_nc=1, ngf=64, netG='deblur_net', device=device1,use_dropout=False,norm='instance')
deblur_net_B.load_state_dict(torch.load(model_path))

for filename in os.listdir(input_dir):
        # Check if the file is a TIFF file
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            file_path = os.path.join(input_dir, filename)
            
            image = imread(file_path)
            #if image has 4 axes then selec the image channel

            if len(image.shape) == 4:
                image = image[args.neuron_ch]

            elif len(image.shape) == 3:
                image = image


            fusion_stack=upsample_block(image,scale,z_step,deblur_net_A,deblur_net_B,min_v,max_v)

            # Optionally, save or further process the image
            out_path = os.path.join(output_dir, filename)
            imwrite(out_path,fusion_stack)



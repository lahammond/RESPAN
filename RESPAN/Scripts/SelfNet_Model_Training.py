# -*- coding: utf-8 -*-
"""

Self-Net Training

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
import numpy as np
from tifffile import imread, imwrite
from skimage.transform import resize
import torch
import torch.nn.functional as F

import random
from torch.utils.data import Dataset
import torch.optim as optim
import itertools

import torch.nn as nn
from torch.nn import init
import functools
import math


'''
The Self-Net code is partially built on Cycle-consistent Generative Adversarial Networks (**CycleGANs**) 
[[paper]](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html), 
which makes unsupervised training of CNNs possible and is really illuminating.
[[code]]https://github.com/junyanz/CycleGAN
'''



# Function to reslice the 3D array
def reslice(data, plane, scale_factor):
    if plane == 'xz':
        
        resliced = np.transpose(data, (0, 1, 2))
        
        #resliced = np.transpose(data, (1, 0, 2))

    elif plane == 'yz':
        resliced = np.transpose(data, (1, 0, 2))
        #resliced = np.transpose(data, (2, 0, 1))
    #resliced = np.array([resize(slice, (m, round(n * scale_factor)), 
     #                          order=3, mode='reflect', preserve_range=True) for slice in resliced])
    
    # Original dimensions
    z, y, x = resliced.shape

    # New dimensions
    new_x = round(x * scale_factor)
    new_dimensions = (z, y, new_x)  # Only scale the 'X' dimension

    # Efficient resizing
    resliced = resize(resliced, new_dimensions, order=3, mode='reflect', preserve_range=True)

    return resliced

def reslice_and_resize_torch(data, plane, scale_factor, device='cuda'):
    # Move data to the specified device (GPU or CPU)
    original_min = np.min(data)
    original_max = np.max(data)
    data = data.astype(np.float32)
    data = torch.from_numpy(data).to(device)

    # Transpose data according to the desired plane
    if plane == 'xz':
        data = data.permute(0, 1, 2)  # Transpose to XZ
    elif plane == 'yz':
        data = data.permute(1, 0, 2)  # Transpose to YZ

    # Original dimensions
    z, y, x = data.shape
    new_x = int(x * scale_factor)

    # Prepare grid for interpolation
    # torch.nn.functional.interpolate requires data in the shape of (batch_size, channels, depth, height, width)
    data = data[None, None, :, :, :]  # Add batch and channel dimensions
    new_size = (z, y, new_x)  # Define new size

    # Resize using interpolate
    resized_data = F.interpolate(data, size=new_size, mode='trilinear', align_corners=False)
    resized_data = resized_data[0, 0]  # Remove batch and channel dimensions

    # Move data back to CPU and convert to numpy if necessary
    return resized_data.cpu().numpy()
    # Normalize back to the original uint16 range
    resized_data = np.clip(resized_data, original_min, original_max)
    resized_data = ((resized_data - original_min) / (original_max - original_min) * 65535).astype(np.uint16)

        
'''   
def tiffread(file_path):
    with tiff.TiffFile(file_path) as tif:
        # Retrieve number of images/frames in the TIFF stack
        number_of_images = len(tif.pages)
        
        # Prepare an empty array to hold the image stack
        # Determine the shape of the first image to pre-allocate the stack array
        sample_image = tif.pages[0].asarray()
        image_shape = sample_image.shape
        stack = np.empty((image_shape[0], image_shape[1], number_of_images), dtype=sample_image.dtype)
        
        # Read each image into the stack
        for i in range(number_of_images):
            stack[:, :, i] = tif.pages[i].asarray()
    
    return stack
'''

def self_net_create_slices(input_dir, filename, scale):
    print(f'Creating training slices for Self-Net... ', flush=True)
    # Read the 3D image data
    #raw_data = tiffread(input_dir)
    raw_data = imread(os.path.join(input_dir, filename))
    print(f' Input image has shape: {raw_data.shape}', flush=True)
    #transpose for selfnet  matlab
    raw_data = np.transpose(raw_data, (1, 2, 0))
    
    m, n, p = raw_data.shape
    
    # Define output directories
    training_slices = os.path.join(input_dir, 'training_slices')
    xy_path = os.path.join(training_slices, 'xy')
    xy_lr_path = os.path.join(training_slices, 'xy_lr')
    xz_path = os.path.join(training_slices, 'xz')
    yz_path = os.path.join(training_slices, 'yz')
    
    
    # Create directories if they don't exist
    for path in [training_slices, xy_path, xy_lr_path, xz_path, yz_path]:
        os.makedirs(path, exist_ok=True)
    
    
    
    # Process xy images
    print(" Creating XY and scaled XY...", flush=True)
    for index in range(p):
        image = raw_data[:, :, index]
    
        
        # Save original image
        imwrite(os.path.join(xy_path, f'{index + 1}.tif'), image, dtype=np.uint16)
    
        # Downsample image
        
        downsampled_image = resize(image, (round(m * scale), n), 
                                   order=3, mode='reflect', preserve_range=True)
        # Upsample the image back to original size
        upsampled_image = resize(downsampled_image, (m, n), 
                                 order=3, mode='reflect', preserve_range=True)
    
        # Save the processed image
        imwrite(os.path.join(xy_lr_path, f'{index + 1}.tif'), upsampled_image.astype(image.dtype))
        
        # Convert float to uint16 directly
        #downsample_img = (downsample_img.clip(0, 65535)).astype(np.uint16)
        #imwrite(os.path.join(xy_lr_path, f'{index + 1}.tif'), downsample_img)
        #tiffwrite(downsample_img, os.path.join(xy_lr_path, f'{index + 1}.tif'))
    print("  XY and scaled XY created.", flush=True)
    
    # Process xz and yz images
    print(" Creating XZ and YZ images...", flush=True)
    for plane, path in [('xz', xz_path), ('yz', yz_path)]:
        resliced_data = reslice_and_resize_torch(raw_data, plane, 1/scale)
        #print(resliced_data.shape)
        new_p, _, _ = resliced_data.shape
    
        for index in range(new_p):
            # Convert float to uint16 directly
            resliced_img = (resliced_data[index, :, : ].clip(0, 65535)).astype(np.uint16)
            resliced_img = np.transpose(resliced_img, (1,0))
            #tiffwrite(resliced_img, os.path.join(path, f'{index + 1}.tif'))
            imwrite(os.path.join(path, f'{index + 1}.tif'), resliced_img)
    print("  YZ and XZ images created. ", flush=True)


def generate_training_data(input_dir, signal_intensity_threshold, xy_interval, xz_interval):
    raw_data_path = input_dir+'training_slices/'
    train_data_path=input_dir+'train_data/'
    
    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)
    
    xy_path = raw_data_path + 'xy/'
    xy_lr_path = raw_data_path + 'xy_lr/'
    xz_path = raw_data_path + 'xz/'
    
    xy = []
    xy_lr = []
    xz = []
    
    stride = 64
    patch_size = 64
    
    
    print("Creating patches for training Self-Net model...", flush=True)
    for i in range(0, len(os.listdir(xy_path)), xy_interval):
    
        xy_img = imread(xy_path + str(i + 1) + '.tif')
        xy_lr_img = imread(xy_lr_path + str(i + 1) + '.tif')
       
    
        for m in range(0, xy_img.shape[0] - patch_size + 1, stride):
            for n in range(0, xy_img.shape[1] - patch_size + 1, stride):
                crop_xy = xy_img[m:m + patch_size, n:n + patch_size]
                crop_xy_lr = xy_lr_img[m:m + patch_size, n:n + patch_size]
    
                if np.max(crop_xy >= signal_intensity_threshold):
                    xy.append(crop_xy)
                    xy_lr.append(crop_xy_lr)
    
    for i in range(0, len(os.listdir(xz_path)), xz_interval):
        xz_img = imread(xz_path + str(i + 1) + '.tif')
        
    
        for m in range(0, xz_img.shape[0] - patch_size + 1, stride):
            for n in range(0, xz_img.shape[1] - patch_size + 1, stride):
                crop_xz = xz_img[m:m + patch_size, n:n + patch_size]
    
                if np.max(crop_xz >= signal_intensity_threshold):
                    xz.append(crop_xz)
    
    xy = np.array(xy, dtype=np.float32)
    xy_lr = np.array(xy_lr, dtype=np.float32)
    xz = np.array(xz, dtype=np.float32)
    print(f' XY = {xy.shape}, XY low res = {xy_lr.shape}, XZ = {xz.shape}', flush=True)
    print(" Patches complete.", flush=True)
    
    np.savez(input_dir + '/train_data/train_data.npz', xy=xy, xy_lr=xy_lr, xz=xz)
    
    
    
###functions for training self-net

def adjust_lr(init_lr,optimizer,epoch,step,gamma):
    if (epoch+1)%step==0:
        times=(epoch+1)/step
        lr=init_lr*gamma**times
        for params in optimizer.param_groups:
            params['lr']=lr


def backward_D_basic(lambda_gan,netD, criterionGAN,real, fake,type,device):
    """Calculate GAN loss for the discriminator
           Parameters:
               netD (network)      -- the discriminator D
               real (tensor array) -- real images
               fake (tensor array) -- images generated by a generator
           Return the discriminator loss.
           We also call loss_D.backward() to calculate the gradients.
           """
    # Real
    pred_real = netD(real)
    loss_D_real = criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake =criterionGAN(pred_fake, False)
    # Combined loss and calculate gradients
    if type=='lsgan' or type=='vanilla':
        loss_D = lambda_gan*(loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    else:
        print('warning in calculating loss_D', flush=True)


def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad


def train(train_set,netD_A,netD_B,netG_A,netG_B,deblur_net, device,criterionGAN,criterionL1,SSIM_loss,optimD,optimG,optim_deblur, epochs, learning_rate_D, learning_rate_G, batch_size, log_interval, imshow_interval, input_dir, ouput_dir, min_v, max_v, num_batches):


    lambda_cycle = 1
    lambda_deblur=1
    lambda_feedback=0
    lambda_gan=0.1
    lambda_ssim=0.1


    netD_A.train()
    netD_B.train()

    netG_A.train()
    netG_B.train()
    deblur_net.train()

    fakeA_pool=ImagePool(50)
    fakeB_pool = ImagePool(50)

    Loss_D=0.0
    Loss_G_GAN=0.0
    Loss_G_cycle=0.0

    Loss_feedback=0.0
    Loss_deblur=0.0


    for epoch in range(epochs):
        param1 = optimD.param_groups[0]
        param2=optimG.param_groups[0]
        adjust_lr(learning_rate_D, optimD, epoch,15,0.5)
        adjust_lr(learning_rate_G, optimG, epoch,15,0.5)
        adjust_lr(learning_rate_G, optim_deblur, epoch,15,0.5)

        print('epoch:{},learning_rate_D:{}'.format(epoch+1,param1['lr']), flush=True)
        print('epoch:{},learning_rate_G:{}'.format(epoch + 1, param2['lr']), flush=True)


        if epoch+1>=2:
            lambda_feedback=0.1


        for i, data in enumerate(train_set):

            #degradation modeling

            A= data['xy_lr']
            B=data['xz']
            C=data['xy']

            A = A.to(device)
            B = B.to(device)
            C=C.to(device)

            fake_B=netG_A(A)      #syn_xz

            fake_A=netG_B(B)

            set_requires_grad(deblur_net,False)
            fake_C1 = deblur_net(fake_B)

            rec_A=netG_B(fake_B)

            rec_B=netG_A(fake_A)

            #optimize G

            set_requires_grad(netD_A, False)
            set_requires_grad(netD_B, False)

            optimG.zero_grad()

            # GAN loss D_A(G_A(A))
            loss_G_A = lambda_gan*criterionGAN(netD_A(fake_B), True)
            # GAN loss D_B(G_B(B))
            loss_G_B = lambda_gan*criterionGAN(netD_B(fake_A), True)


            # Forward cycle loss || G_B(G_A(A)) - A||


            loss_cycle_A = lambda_cycle*criterionL1(rec_A,A)+lambda_ssim*(1-SSIM_loss(rec_A,A))

            # Backward cycle loss || G_A(G_B(B)) - B||

            loss_cycle_B = lambda_cycle*criterionL1(rec_B,B)+lambda_ssim*(1-SSIM_loss(rec_B,B))


            loss_feedback = lambda_feedback*(criterionL1(fake_C1, C)+lambda_ssim*(1-SSIM_loss(fake_C1,C)))

            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B+loss_feedback

            loss_G.backward()
            optimG.step()

            #optimize D

            if (i+1)%2==0:
                set_requires_grad(netD_A, True)
                set_requires_grad(netD_B, True)

                optimD.zero_grad()

                fake_B = fakeB_pool.query(fake_B)

                loss_D_A =backward_D_basic(lambda_gan,netD_A, criterionGAN, B, fake_B, 'lsgan', device)

                fake_A = fakeA_pool.query(fake_A)
                loss_D_B = backward_D_basic(lambda_gan,netD_B,criterionGAN,A,fake_A,'lsgan',device)

                optimD.step()

            else:
                loss_D_A=torch.Tensor([0.0])
                loss_D_B=torch.Tensor([0.0])

            #Checkpoint

            Loss_D += loss_D_A.item() + loss_D_B.item()
            Loss_G_GAN+=loss_G_A.item()+loss_G_B.item()

            Loss_feedback+=loss_feedback.item()

            Loss_G_cycle+=loss_cycle_A.item()+loss_cycle_B.item()

            if (i + 1) % log_interval == 0:
                print('epochs: {} {}/{}  degradation modeling stage: loss_D:{:4f} , '
                      'loss_G: {:4f} loss_G_cycle:{:4f} loss_feedback:{:4f}'.
                format(epoch+1,

                (i + 1) * batch_size, len(train_set.dataset),
                Loss_D / (log_interval*lambda_gan),Loss_G_GAN/(log_interval*lambda_gan),
                Loss_G_cycle / (log_interval*lambda_cycle),Loss_feedback/(log_interval)), flush=True)

                Loss_D = 0.0
                Loss_G_GAN = 0.0
                Loss_G_cycle = 0.0
                Loss_feedback=0.0

            #deblurring

            set_requires_grad(deblur_net, True)
            fake_B = netG_A(A)  # syn_lr_xy
            fake_C2 = deblur_net(fake_B.detach())

            optim_deblur.zero_grad()

            loss_deblur = lambda_deblur * criterionL1(fake_C2, C) + lambda_ssim * (1 - SSIM_loss(fake_C2, C))

            loss_deblur.backward()
            optim_deblur.step()

            # Checkpoint

            Loss_deblur += loss_deblur.item()

            if (i + 1) % log_interval == 0:
                print('epochs: {} {}/{}  deblurring stage :loss_deblur:{:4f}'.
                      format(epoch + 1,
                             (i + 1) * batch_size, len(train_set.dataset),Loss_deblur /(log_interval*lambda_deblur)), flush=True)

                Loss_deblur = 0.0

           # if (i + 1) % imshow_interval == 0:
            #    intermediate_output(deblur_net, device, chkpoint_dir, min_v, max_v, epoch, (i+1)*batch_size)
            #    torch.save(deblur_net.state_dict(),chkpoint_dir+'/checkpoint/saved_models/' + 'deblur_net_{}_{}'.format (epoch + 1,(i+1)*batch_size) + '.pkl')

            if (i + 1) % imshow_interval == 0 or (i + 1) == num_batches:
                print('Saving intermediate results and model snapshot... ', flush=True)
                # visual slice

                intermediate_output(deblur_net, device, input_dir, ouput_dir,
                                    min_v, max_v, epoch, (i + 1) * batch_size)

                # model snapshot
                save_dir = os.path.join(ouput_dir, 'saved_models')
                os.makedirs(save_dir, exist_ok=True)  # guarantees path exists
                torch.save(
                    deblur_net.state_dict(),
                    os.path.join(
                        save_dir,
                        f'SelfNet_Epoch_{epoch + 1}_Batch_{(i + 1) * batch_size}.pkl'  # one “checkpoint” in path only
                    )
                )

###############################################################################
##### Helper functions for training
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

    print('initialize network with %s' % init_type, flush=True)
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


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blured tensors
    """
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y,
          data_range,
          win,
          size_average=True,
          K=(0.01, 0.03)):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(X, Y,
         data_range=255,
         size_average=True,
         win_size=11,
         win_sigma=1.5,
         win=None,
         K=(0.01, 0.03),
         nonnegative_ssim=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same shape.')

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)

    ssim_per_channel, cs = _ssim(X, Y,
                                 data_range=data_range,
                                 win=win,
                                 size_average=False,
                                 K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(X, Y,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            win=None,
            weights=None,
            K=(0.01, 0.03)):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same dimensions.')

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4), \
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y,
                                     win=win,
                                     data_range=data_range,
                                     size_average=False,
                                     K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = (X.shape[2] % 2, X.shape[3] % 2)
            X = F.avg_pool2d(X, kernel_size=2, padding=padding)
            Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(self,
                 data_range=255,
                 size_average=True,
                 win_size=11,
                 win_sigma=1.5,
                 channel=3,
                 K=(0.01, 0.03),
                 nonnegative_ssim=False):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(X, Y,
                    data_range=self.data_range,
                    size_average=self.size_average,
                    win=self.win,
                    K=self.K,
                    nonnegative_ssim=self.nonnegative_ssim)


class MS_SSIM(torch.nn.Module):
    def __init__(self,
                 data_range=255,
                 size_average=True,
                 win_size=11,
                 win_sigma=1.5,
                 channel=3,
                 weights=None,
                 K=(0.01, 0.03)):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(X, Y,
                       data_range=self.data_range,
                       size_average=self.size_average,
                       win=self.win,
                       weights=self.weights,
                       K=self.K)

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
    
def intermediate_output(
        deblur_net,
        device,
        input_dir,
        model_root,# <- pass the *model root* not the checkpoint
        min_v, max_v,
        epoch, iters,
        slice_idx=1):

    deblur_net.eval()
    cpu = torch.device('cpu')

    # 1.  intermediate_results directory  (no double “checkpoint”)
    vis_dir = os.path.join(model_root, 'intermediate_results')
    os.makedirs(vis_dir, exist_ok=True)

    train_slices = os.path.join(input_dir, 'training_slices')
    #os.makedirs(train_slices, exist_ok=True)

    # 2.  location of the slice to visualise
    slice_path = os.path.join(
        train_slices, 'xz', f'{slice_idx}.tif')

    if not os.path.isfile(slice_path):
        print(f'   preview slice not found: {slice_path}')
        deblur_net.train()
        return                     # fail gracefully instead of crashing

    # 3.  read & normalise
    img = imread(slice_path).astype(np.float32)
    img = np.clip((img - min_v) / (max_v - min_v), 0, 1)
    img = img[None, None, ...]                   # (1,1,H,W)
    with torch.no_grad():
        out = deblur_net(torch.from_numpy(img).to(device))

    out = out.squeeze().to(cpu).numpy()          # H,W
    out = np.clip(out * (max_v - min_v) + min_v, 0, max_v).astype(np.uint16)

    # 4.  save
    fname = f'SelfNet_epoch{epoch+1:03d}_iters{iters}_{slice_idx}.tif'
    imwrite(os.path.join(vis_dir, fname), out)

    deblur_net.train()
    
###### pytorch_dataset functions

def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 3:
        return np.rot90(img, k=2)
    elif mode == 4:
        return np.flipud(np.rot90(img))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def transform_input_img(img,mode,min_v,max_v):
    img = augment_img(img,mode)
    img=np.array(img,dtype=np.float32)
    img=(img-min_v)/(max_v-min_v)
    img[img>1]=1
    img[img<0]=0
    img=np.expand_dims(img,axis=0)
    return torch.from_numpy(img)

def transform_target_img(img,mode,min_v,max_v):
    img=augment_img(img,mode)
    img=np.array(img,dtype=np.float32)
    img = (img-min_v)/ (max_v-min_v)
    img[img>1]=1
    img[img < 0] = 0
    img=np.expand_dims(img,axis=0)
    return torch.from_numpy(img)


class ImageDataset_numpy(Dataset):
    def __init__(self,dataset,paired,min_v,max_v):
        self.xy=dataset['xy']
        self.xy_lr=dataset['xy_lr']
        self.xz=dataset['xz']
        self.paired=paired
        self.min_v=min_v
        self.max_v=max_v

    def __getitem__(self, index):

        mode1=np.random.randint(0, 8)
        mode2=np.random.randint(0, 8)
        if self.paired:
            item_xz = transform_input_img(self.xz[(index)%(self.xz.shape[0])], mode1,self.min_v,self.max_v)
            item_xy = transform_target_img(self.xy[(index) % (self.xy.shape[0])], mode1,self.min_v,self.max_v)
            item_xy_lr = transform_target_img(self.xy_lr[(index) % (self.xy_lr.shape[0])], mode1,self.min_v,self.max_v)
        else:
            item_xz = transform_input_img(self.xz[random.randint(0, (self.xz).shape[0]- 1)], mode1,self.min_v,self.max_v)
            item_xy = transform_target_img(self.xy[(index)%(self.xy.shape[0])], mode2,self.min_v,self.max_v)
            item_xy_lr=transform_target_img(self.xy_lr[(index)%(self.xy_lr.shape[0])], mode2,self.min_v,self.max_v)

        return {'xz': item_xz, 'xy': item_xy, 'xy_lr':item_xy_lr}

    def __len__(self):
        return max((self.xz).shape[0], (self.xy).shape[0])


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

def create_train_data(train_data_path,batch_size,min_v,max_v):

    dataset=np.load(train_data_path+'train_data.npz')
    train_data=torch.utils.data.DataLoader(ImageDataset_numpy(dataset,False,min_v,max_v),batch_size=batch_size,shuffle=True)

    print('Training data created.', flush=True)
    return train_data


def train_self_net(input_dir, out_path, min_v, max_v, batch_size=8, epochs=60, learning_rate_G=1e-4, learning_rate_D=1e-4, beta1=0.5,
                   beta2=0.999, log_interval=1, imshow_interval=100):


    model_saved_path = os.path.join(out_path, 'saved_models')
    intermediate_results_path = os.path.join(out_path, 'intermediate_results')

    if not os.path.exists(model_saved_path):
        os.mkdir(model_saved_path)
    if not os.path.exists(intermediate_results_path):
        os.mkdir(intermediate_results_path)

    device = torch.device('cuda:0')

    input_nc = 1
    output_nc = 1

    # Define the networks
    netD_A = define_D(input_nc=input_nc, ndf=64, netD='n_layers', n_layers_D=2, device=device, norm='instance')
    netD_B = define_D(input_nc=input_nc, ndf=64, netD='n_layers', n_layers_D=2, device=device, norm='instance')
    netG_A = define_G(input_nc=input_nc, output_nc=output_nc, ngf=64, netG='deblur_net', device=device,
                      use_dropout=False, norm='instance')
    netG_B = define_G(input_nc=input_nc, output_nc=output_nc, ngf=64, netG='deblur_net', device=device,
                      use_dropout=False, norm='instance')
    deblur_net = define_G(input_nc=input_nc, output_nc=output_nc, ngf=64, netG='deblur_net', device=device,
                          use_dropout=False, norm='instance')

    # Define the losses
    criterionGAN = GANLoss(gan_mode='lsgan').to(device)
    SSIM_loss = SSIM(data_range=1, size_average=True, win_size=11, win_sigma=1.5, channel=1)
    criterionL1 = L1_Charbonnier_loss()

    # Setup optimizers
    optimG = optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr=learning_rate_G,
                        betas=(beta1, beta2))
    optimD = optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=learning_rate_D,
                        betas=(beta1, beta2))
    optim_deblur = optim.Adam(deblur_net.parameters(), lr=learning_rate_G, betas=(beta1, beta2))

    # Load and train the dataset
    train_set = create_train_data(input_dir + 'train_data/', batch_size, min_v, max_v)

    num_batches = len(train_set)
    imshow_interval = min(imshow_interval, num_batches)

    train(train_set, netD_A, netD_B, netG_A, netG_B, deblur_net, device, criterionGAN, criterionL1, SSIM_loss, optimD, optimG, optim_deblur,
          epochs, learning_rate_D, learning_rate_G, batch_size, log_interval, imshow_interval, input_dir, out_path, min_v, max_v,num_batches)
    print('Training complete.', flush=True)

###PART 1: Create Slices


import argparse
# Create the parser

def parse_args():
    parser = argparse.ArgumentParser(description="Train Self-Net Model")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory for input data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the model')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--min_v', type=int, default=0, help='Minimum intensity value')
    parser.add_argument('--max_v', type=int, default=65535, help='Maximum intensity value')
    parser.add_argument('--bg_threshold', type=int, default=1000, help='Background threshold for patch selection')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale factor for XY to Z')
    parser.add_argument('--xy_int', type=int, default=4, help='Interval in XY plane')
    parser.add_argument('--xz_int', type=int, default=8, help='Interval in XZ plane')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs for training')
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval for training')
    parser.add_argument('--imshow_interval', type=int, default=100, help='Image show interval for training')

    # Add other parameters here as needed
    return parser.parse_args()

def main():
    args = parse_args()

    print(args, flush=True)

    #  use the arguments
    input_dir = args.input_dir
    model_path = args.model_path
    model_name = args.model_name
    min_v = args.min_v
    max_v = args.max_v
    bg_threshold = args.bg_threshold
    scale = args.scale
    xy_interval = args.xy_int
    xz_interval = args.xz_int
    batch_size = args.batch_size
    epochs = args.epochs
    log_interval = args.log_interval
    imshow_interval = args.imshow_interval

    output_dir = os.path.join(model_path, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set paths and file name
    # input_dir = 'D:/Project_Data/RESPAN/_Axial_restoration/SelfnetV2/Example_raw_in/'
    # filename = '60Xthy_confocal_4.tif'

    # Scaling factor
    # scale = 0.21

    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    if tif_files:
        filename = tif_files[0]  # Select the first .tif file
    else:
        filename = None  # No .tif files found

    # Proceed only if a .tif file was found
    if filename:
        self_net_create_slices(input_dir, filename, scale)
    else:
        print("No .tif files found in the specified directory.", flush=True)

    ##################################################

    ###PART 2:  Generate Training Data
    # Input parameters: path, raw_data_path, train_data_path, signal_intensity_threshold, xy_interval, xz_interval

    # signal_intensity_threshold=600  #min threshold parameter for selecting image patches containing signal
    # xy_interval=4
    # xz_interval=8 #reduced from 12 to match number of patches based on sn paper

    generate_training_data(input_dir, bg_threshold, xy_interval, xz_interval)

    ######################################################

    ##PART 3:  Train Self-Net model

    # Input parameters:  path, min_v, max_v, imshow_interval
    # min_v = 0
    # max_v = 65535 #default was 4095 (12bit images)

    train_self_net(
        input_dir,  # Path where the training data is stored
        output_dir,
        min_v,  # Minimum intensity value
        max_v,  # Maximum intensity value
        batch_size=batch_size,  # Batch size
        epochs=epochs,  # Number of epochs
        learning_rate_G=1e-4,
        learning_rate_D=1e-4,
        beta1=0.5,
        beta2=0.999,
        log_interval=log_interval,
        imshow_interval=imshow_interval
    )


if __name__ == '__main__':
    main()




"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from networks.discriminator import *
from networks.loss import *
import networks.denoising_raw

def define_D():
    netD_cls = discriminator.NLayerDiscriminator()
    netD_cls.print_network()
    return netD_cls

def define_G(in_channels):
    netG_cls  = denoising_raw.DenoiseNet(in_channels)
    return netG_cls
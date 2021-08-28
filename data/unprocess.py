# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unprocesses sRGB images into realistic raw data.
Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
from scipy.stats import poisson

def generate_noisy_raw(gt_raw, a, b):
    """
    a: sigma_s^2
    b: sigma_r^2
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw-240)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    noisy_img = poisson_noisy_img + gaussian_noise + 240
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)

    return noisy_img

def generate_ccm():
  """Generates random RGB -> Camera color correction matrices."""
  # Takes a random convex combination of XYZ -> Camera CCMs.
  cam2xyz = [[0.4679,0.2145,0.3176],[0.1433,0.8236,0.0331],[0.0003,-0.3607,1.3604]]
  xyz2cam = torch.from_numpy(np.linalg.inv(np.array(cam2xyz)).astype(np.float32))

  # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
  rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                               [0.2126729, 0.7151522, 0.0721750],
                               [0.0193339, 0.1191920, 0.9503041]])
  rgb2cam = torch.mm(xyz2cam, rgb2xyz)

  # Normalizes each row.
  rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
  cam2rgb = torch.inverse(rgb2cam)
  return rgb2cam, cam2rgb


def random_gains():
  """Generates random gains for brightening and white balance."""
  # RGB gain represents brightening.
  rgb_gain = 1.0

  # Red and blue gains represent white balance.
  red_gain  =  torch.FloatTensor([1.0 / 0.5527])
  blue_gain =  torch.FloatTensor([1.0 / 0.4844])
  return rgb_gain, red_gain, blue_gain


def inverse_smoothstep(image):
  """Approximately inverts a global tone mapping curve."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  image = torch.clamp(image, min=0.0, max=1.0)
  out   = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0) 
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def gamma_expansion(image):
  """Converts from gamma to linear space."""
  # Clamps to prevent numerical instability of gradients near zero.
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  out   = torch.clamp(image, min=1e-8) ** 2.2
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def apply_ccm(image, ccm):
  """Applies a color correction matrix."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  shape = image.size()
  image = torch.reshape(image, [-1, 3])
  image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
  out   = torch.reshape(image, shape)
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
  """Inverts gains while safely handling saturated pixels."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  gains = torch.stack((1.0 / red_gain, torch.tensor([1.0]), 1.0 / blue_gain)) / rgb_gain
  gains = gains.squeeze()
  gains = gains[None, None, :]
  # Prevents dimming of saturated pixels by smoothly masking gains near white.
  gray  = torch.mean(image, dim=-1, keepdim=True)
  inflection = 0.9
  mask  = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
  safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
  out   = image * safe_gains
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def mosaic(image):
  """Extracts RGGB Bayer planes from an RGB image."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  shape = image.size()
  red = image[1::2, 0::2, 0]
  green_red = image[1::2, 1::2, 1]
  green_blue = image[0::2, 0::2, 1]
  blue = image[0::2, 1::2, 2]
  out  = torch.stack((green_blue, blue, red, green_red), dim=-1)
  out  = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
  out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out

def metadata(rgb2cam, cam2rgb):
  """Randomly creates image metadata."""
  rgb_gain, red_gain, blue_gain = random_gains()
  
  metadata = {
      'rgb2cam': rgb2cam,
      'cam2rgb': cam2rgb,
      'rgb_gain': rgb_gain,
      'red_gain': red_gain,
      'blue_gain': blue_gain,
  }
  return metadata

def depack_gbrg_raw(raw):
    """Depack packed raw to generate GBRG Bayer raw."""
    black_level = 240
    white_level = 2**12-1

    raw = raw*(white_level-black_level)+black_level

    H = raw.shape[0]
    W = raw.shape[1]
    output = np.zeros((H*2,W*2))

    output[ ::2, ::2]=raw[...,0]
    output[ ::2,1::2]=raw[...,1]
    output[1::2, ::2]=raw[...,2]
    output[1::2,1::2]=raw[...,3]
    return output

def unprocess(image, metadata):
  rgb2cam   = metadata['rgb2cam']
  rgb_gain  = metadata['rgb_gain']
  red_gain  = metadata['red_gain']
  blue_gain = metadata['blue_gain']
  
  """Unprocesses an image from sRGB to realistic raw data."""
  # Approximately inverts global tone mapping.
  image = inverse_smoothstep(image)
  # Inverts gamma compression.
  image = gamma_expansion(image)
  # Inverts color correction.
  image = apply_ccm(image, rgb2cam)
  # Approximately inverts white balance and brightening.
  image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
  # Clips saturated pixels.
  image = torch.clamp(image, min=0.0, max=1.0)
  # Applies a Bayer mosaic.
  image = mosaic(image)

  return image
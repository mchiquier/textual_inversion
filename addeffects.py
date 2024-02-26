import numpy as np
import cv2 as cv

import cv2 
import math
import torch

from matplotlib import pyplot as plt
from PIL import Image
import copy 
import os
import subprocess
import json
import imgaug.augmenters as iaa
import os
import pdb
import sys

import PIL

from numpy.fft import fft2, ifft2, fftshift, ifftshift
from torch.nn import functional as F
from math import pi


"""My helpers"""

def read_image_resize(current_img_path, resize=(0, 0)):
   
    im = Image.open(current_img_path)

    if resize[0] != 0:
        img = resample_lanczos(im, resize[0], resize[1])
    else:
        img = np.array(im, dtype=np.uint8)

    return img

def resample_lanczos(im: PIL.Image, W: int, H: int) -> np.ndarray:
    """Resize the image correctly. Thanks to Jaakko Lehtinin for the tip."""
    new_size = (W, H)
    im = im.resize(new_size, Image.LANCZOS)

    return np.array(im)

def affine_transform(
    img: torch.Tensor,
    affine_matrices: torch.Tensor,
    scale=True,
    translate=True,
    rotate=True,
) -> torch.Tensor:
    """Input arguments:
    SUPER IMPORTANT FOR IMG TO BE CHANNEL FIRST COS AFFINE_GRID 
    WILL DO ITS THING REGARLDESS WITHOUT THROWING AN ERROR
    img: b x c x w x h tensor or c x w x h tensor
    affine_matrices: 2x3 tensor
    Returns: transformed_img: bxcxwxh"""

    if len(img.shape) == 3:
        img = torch.unsqueeze(img, dim=0)

    try:
        assert img.shape[1] < 5
    except AssertionError:
        raise Exception("img not channel first")

    grid = F.affine_grid(affine_matrices, list(img.shape))
    transformed_image = F.grid_sample(img, grid, mode="bilinear")

    return transformed_image

"""Display original image"""

"""3. Rain/clouds/snow: I switched fog for clouds because clouds is less uniform"""


seed=2

def saturate_easy(theimg):
  image = PIL.Image.fromarray(theimg[0])
  new_image_c = PIL.ImageEnhance.Color(image).enhance(1.5)
  cv2.imwrite("sateasy.png",cv2.cvtColor(np.asarray(new_image_c), cv2.COLOR_RGB2BGR))
  #plt.figure()
  #plt.imshow(np.asarray(new_image_c))
  #plt.title("Color easy"), plt.xticks([]), plt.yticks([])

#saturate_easy(theimg)

def saturate_medium(theimg):
  image = PIL.Image.fromarray(theimg[0])
  new_image_c = PIL.ImageEnhance.Color(image).enhance(3)
  cv2.imwrite("satmedium.png",cv2.cvtColor(np.asarray(new_image_c), cv2.COLOR_RGB2BGR))
  #plt.figure()
  #plt.imshow(np.asarray(new_image_c))
  #plt.title("Color medium"), plt.xticks([]), plt.yticks([])

#saturate_medium(theimg)

def saturate_hard(theimg):
  image = PIL.Image.fromarray(theimg[0])
  new_image_c = PIL.ImageEnhance.Color(image).enhance(4.5)
  cv2.imwrite("sathard.png",cv2.cvtColor(np.asarray(new_image_c), cv2.COLOR_RGB2BGR))
  #plt.figure()
  #plt.imshow(np.asarray(new_image_c))
  #plt.title("Color hard"), plt.xticks([]), plt.yticks([])

#saturate_hard(theimg)

def add_snow_easy(theimg, seed):
  aug_clouds = iaa.CloudLayer(intensity_mean=200, intensity_freq_exponent=-1.5, intensity_coarse_scale=10, 
  alpha_min=0.0,alpha_multiplier=1.0, alpha_size_px_max=1300, alpha_freq_exponent=-4, 
  sparsity=0.3, density_multiplier=0.5, seed=seed)
  img_aug_clouds = aug_clouds(images=theimg)
  plt.figure()
  plt.imshow(img_aug_clouds[0])
  #plt.title("Clouds easy"), plt.xticks([]), plt.yticks([])
  cv2.imwrite("cloudseasy.png",cv2.cvtColor(img_aug_clouds[0], cv2.COLOR_RGB2BGR))
  #plt.savefig("cloudseasy.png")

#add_snow_easy(theimg, seed)

def add_snow_medium(theimg, seed):
  aug_clouds = iaa.CloudLayer(intensity_mean=200, intensity_freq_exponent=-1.5, intensity_coarse_scale=10, 
  alpha_min=0.0,alpha_multiplier=1.0, alpha_size_px_max=1300, alpha_freq_exponent=-4, 
  sparsity=0.3, density_multiplier=0.75, seed=seed)
  img_aug_clouds = aug_clouds(images=theimg)
  cv2.imwrite("cloudsmed.png",cv2.cvtColor(img_aug_clouds[0], cv2.COLOR_RGB2BGR))

#add_snow_medium(theimg, seed)

def add_snow_hard(theimg, seed, filename):
  aug_clouds = iaa.CloudLayer(intensity_mean=200, intensity_freq_exponent=-1.5, intensity_coarse_scale=10, 
  alpha_min=0.0,alpha_multiplier=1.0, alpha_size_px_max=1300, alpha_freq_exponent=-4, 
  sparsity=0.3, density_multiplier=1.0, seed=seed)
  img_aug_clouds = aug_clouds(images=theimg)

  cv2.imwrite("destdirclouds/" + filename,cv2.cvtColor(img_aug_clouds[0], cv2.COLOR_RGB2BGR))

imgs = os.listdir("destdir")
for file in imgs:
    theimg = np.expand_dims(read_image_resize("destdir/" + file),axis=0)
    add_snow_hard(theimg, seed, file)
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from PIL import Image
import os
import random
import numpy as np
import cv2 as cv

from data import unprocess
from data.utils import pack_gbrg_raw


def _calc_soft_gradient_mask(imageRGB):
    """
    Calculate soft gradient mask of the GT image.

    Parameters
    ----------
        dir : PIL.Image
            ground truth RGB PIL image.
    Returns
    -------
        np.array
            soft gradient mask.
    """
    w = 3          # window size is WxW
    thr = 0.2      # threshold
    
    img = np.array(imageRGB.convert('L')).astype(np.float32) / 255.0

    imgDiffX = cv.Sobel(img, cv.CV_32F, 1, 0, 3)
    imgDiffY = cv.Sobel(img, cv.CV_32F, 0, 1, 3)
    
    imgDiffXX = cv.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv.multiply(imgDiffY, imgDiffY)
    J11 = cv.boxFilter(imgDiffXX, cv.CV_32F, (w,w))
    J22 = cv.boxFilter(imgDiffYY, cv.CV_32F, (w,w))

    J11[J11 < 0] = 0
    J22[J22 < 0] = 0
    lambda1 = np.sqrt(J11 + J22) / np.sqrt(32)
    #soft mask
    mask = np.tanh(lambda1/thr)

    return mask

def _make_dataset(dir):
    """
    Creates a 2D list of all the frames in N clips containing
    M frames each.
    2D List Structure:
    [[frame00, frame01,...frameM]  <-- clip0
     [frame00, frame01,...frameM]  <-- clip0
     :
     [frame00, frame01,...frameM]] <-- clipN
    Parameters
    ----------
        dir : string
            root directory containing clips.
    Returns
    -------
        list
            2D list described above.
    """

    framesPath = []
    # Find and loop over all the clips in root `dir`.
    for folder in os.listdir(dir):
        subFolderPath = os.path.join(dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(subFolderPath)):
            continue
        
        for subFolder in os.listdir(subFolderPath):
            clipsFolderPath = os.path.join(subFolderPath, subFolder)
            # Skip items which are not folders.
            if not (os.path.isdir(clipsFolderPath)):
                continue
            
            clips = []
            # Find and loop over all the frames inside the clip.
            for image in sorted(os.listdir(clipsFolderPath)):
                # Add path to list.
                clips.append(os.path.join(clipsFolderPath, image))

            framesPath.append(clips)

    return framesPath

def _pil_loader(path):
    """
    Opens image at `path` using pil.

    Parameters
    ----------
        path : string
            path of the image.

    Returns
    -------
        PIL.Image
            PIL Image object.
    """


    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def _transforms(images, size, train):
    """
    Apply same transformation on all images in the
    sequence.

    Parameters
    ----------
        images : list
            list of images in the sequence.
        size : tuple
            Cropping dimensions
        train : boolean, optional
            Specifies if the sequence is for training or testing/validation.
            `True` returns samples with data augmentation like random 
            flipping, random cropping, etc. while `False` returns the
            samples without randomization.
        

    Returns
    -------
        list
            List of transformed images.
    """

    # Init
    i, j, h, w = transforms.RandomCrop.get_params(
        images[0], output_size=size) if train else (0, 0, size[1], size[0])
    randHorizontalFlip = random.random() > 0.5
    randVerticalFlip = random.random() > 0.5
    returnImages = []
    for image in images:
        # Random/Fixed (train/val) crop
        image = TF.crop(image, i, j, h, w) 
        if train:
            # Random horizontal flipping
            if randHorizontalFlip:
                image = TF.hflip(image)

            # Random vertical flipping
            if randVerticalFlip:
                image = TF.vflip(image)

        # Transform to tensor
        returnImages.append(TF.to_tensor(image))

    return returnImages


class VideoDenoiseDataloader(data.Dataset):
    """
    A dataloader for loading N samples arranged in this way:

        |-- clip0
            |-- frame00
            |-- frame01
            :
            |-- frame11
            |-- frame12
        |-- clip1
            |-- frame00
            |-- frame01
            :
            |-- frame11
            |-- frame12
        :
        :
        |-- clipN
            |-- frame00
            |-- frame01
            :
            |-- frame11
            |-- frame12
    ...
    Attributes
    ----------
    framesPath : list
        List of frames' path in the dataset.
    Methods
    -------
    __getitem__(index)
        Returns the sample corresponding to `index` from dataset.
    __len__()
        Returns the size of dataset. Invoked as len(datasetObj).
    __repr__()
        Returns printable representation of the dataset object.
    """


    def __init__(self, root, sequenceLength, randomCropSize, train=True):
        """
        Parameters
        ----------
            root : string
                Root directory path.
            sequenceLength : int
                Number of  images in input sequence.
            dim : tuple, optional
                Dimensions of images in dataset. Default: (448, 256)
            randomCropSize : tuple, optional
                Dimensions of random crop to be applied. Default: (192, 192)
            train : boolean, optional
                Specifies if the dataset is for training or testing/validation.
                `True` returns samples with data augmentation like random 
                flipping, random cropping, etc. while `False` returns the
                samples without randomization. Default: True
        """


        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath = _make_dataset(root)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
        
        self.sequenceLength = sequenceLength
        self.randomCropSize = randomCropSize
        self.root           = root
        self.train          = train
        self.framesPath     = framesPath
        self.clipLength = len(self.framesPath[0])
        self.iso_list = [1600,3200,6400,12800,25600]
        self.a_list = [3.513262,6.955588,13.486051,26.585953,52.032536]
        self.b_list = [11.917691,38.117816,130.818508,484.539790,1819.818657]
        self.noiseLevels = len(self.iso_list)
        self.rgb2cam, self.cam2rgb = unprocess.generate_ccm()
        self.ToPILImage = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Returns the sample corresponding to `index` from dataset.
        The sample consists of noisy image sequence and variance.
        Parameters
        ----------
            index : int
                Index
        Returns
        -------
            tuple
                (noisy_img_seq, GT_raw, metadata, gan_mask) where 
                noisy_img_seq is the noisy image sequence + 
                variance tensor, GT_raw is the ground truth raw 
                image, synthetic random camera metadata and
                gan_mask is the soft gradient mask.
        """


        if (self.train):
            # Select random m contiguous frames from N frames in a clip
            firstFrame = random.randint(0, self.clipLength - self.sequenceLength)
            # Random reverse frame
            frameRange = range(firstFrame, firstFrame + self.sequenceLength) \
                if (random.randint(0, 1)) else \
                    range(firstFrame + self.sequenceLength - 1, firstFrame - 1, -1)
            # Random noise index
            noiseIndex = np.random.randint(0, self.noiseLevels)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            noiseIndex = index % self.noiseLevels
            frameRange = [0, 1, 3, 5, 7]
        
        # Open image using pil and augment the image.
        imageSeq = [_pil_loader(self.framesPath[index][f]) for f in frameRange]
        metadata  = unprocess.metadata(self.rgb2cam, self.cam2rgb)
        shot_noise, read_noise = self.a_list[noiseIndex], self.b_list[noiseIndex]
        
        # Apply transformations and unprocess pipeline
        imageTSeq  = _transforms(imageSeq, self.randomCropSize, train=self.train)
        imageTUSeq = [unprocess.unprocess(imageT, metadata).permute(1, 2, 0) for imageT in imageTSeq]
        imageTUTSeq = [unprocess.depack_gbrg_raw(imageTU.numpy()) for imageTU in imageTUSeq]

        # Generate GAN Mask for sharp features
        gan_mask = _calc_soft_gradient_mask(self.ToPILImage(imageTSeq[self.sequenceLength//2]))

        # Add random noise
        imageTUTNSeq = [unprocess.generate_noisy_raw(imageTUT, shot_noise, read_noise) for imageTUT in imageTUTSeq]

        # Pack input and GT to RGBG
        imageTUTNPSeq = [pack_gbrg_raw(imageTUTN) for imageTUTN in imageTUTNSeq]
        imageTUTPSeq  = [pack_gbrg_raw(imageTUT)  for imageTUT  in imageTUTSeq]

        noisy_img_seq = torch.cat([torch.from_numpy(x).permute(2, 0, 1) for x in imageTUTNPSeq], 0)
        GT_raw        = torch.cat([torch.from_numpy(x).permute(2, 0, 1) for x in imageTUTPSeq],  0)
        
        return noisy_img_seq, GT_raw, metadata, gan_mask


    def __len__(self):
        """
        Returns the size of dataset. Invoked as len(datasetObj).
        Returns
        -------
            int
                number of samples.
        """


        return len(self.framesPath)

    def __repr__(self):
        """
        Returns printable representation of the dataset object.
        Returns
        -------
            string
                info.
        """


        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """


    def __init__(self, dim, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(backWarp, self).__init__()
        # create a grid
        W, H = int(dim[0] / 2), int(dim[1] / 2)
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False).to(device)
        self.gridY = torch.tensor(gridY, requires_grad=False).to(device)
        
    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut
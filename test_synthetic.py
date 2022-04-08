import torch

import os
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
import cv2
from time import time

from data import dataloader
from networks import denoising_raw
from core.raft import RAFT
from data.utils import *

iso_list = ['25600', '12800', '6400', '3200', '1600']

def forward(input):
    # Denoising function
    input_raw_left, input_raw_centr, input_raw_right = input

    with torch.no_grad():
        # Calculate optical flows - RAFT
        flow_left, context_centr = flowComp(input_raw_centr, input_raw_left, 32)
        flow_right, _ = flowComp(input_raw_centr, input_raw_right, 32)

        input_raw_left  = backWarp(input_raw_left, flow_left[-1])
        input_raw_right = backWarp(input_raw_right, flow_right[-1])

        # Denoise
        input_raw = torch.cat((input_raw_left, input_raw_centr, input_raw_right, context_centr), 1)
        out_raw = input_raw_centr + denoiser(input_raw)

    return out_raw

def denoise_syn(data_dir, output_dir, example):
    # Denoising function for synthetic videos

    allImageNames = os.listdir(data_dir)
    allImageNames.sort()
    numOfFrames = 5
    counter = 0

    frame_avg_raw_psnr = 0
    frame_avg_raw_ssim = 0
    frame_avg_srgb_psnr = 0
    frame_avg_srgb_ssim = 0
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # List to store previous overlapping denoised frames
    prev = [None]

    with torch.no_grad():
        for index in tqdm(range(len(allImageNames) - numOfFrames + 1)):
            imagesNames = [allImageNames[ind] for ind in range(index, index + numOfFrames)]
            images = []

            # Denoise each image.
            # start = time.time()
            for imageName in imagesNames:
                # Loads the noisy image.
                image_raw = Image.open(os.path.join(data_dir, imageName))
                image_raw = pack_gbrg_raw(np.array(image_raw))
                image_raw = torch.from_numpy(image_raw).permute(2, 0, 1).cuda().unsqueeze(0)
                images.append(image_raw)

            temp = []

            for i in range(3):
                # Check if previous denoised frames exist for overlapping sequences
                temp.append((prev[-1][i+1]) if (prev[-1] is not None and (i+1) < 3) else forward(images[i:i+3]))
            prev[-1] = temp

            output = forward(prev[-1])

            # Padded output cropped to 1920 x 1080 from 1920x1952
            # Reflection padding showed slightly better results
            output = output.cpu().detach().permute(0, 2, 3, 1)
            output = depack_gbrg_raw(output)[36:-36]

            if args.raw_psnr:
                ### RAW PSNR
                test_gt = cv2.imread(args.input_dir + '/gt_raw/{}/{}'.format(example, imagesNames[int(numOfFrames/2)]),-1).astype(np.float32)[36:-36, :]
                test_gt = (test_gt-240)/(2**12-1-240)

                test_raw_psnr = compare_psnr(test_gt, output, data_range=1.0)
                test_raw_ssim = compute_ssim_for_packed_raw(test_gt, output)
                context = 'scene {} frame{} test raw psnr : {}, test raw ssim : {} '.format(example, imagesNames[int(numOfFrames/2)].split('.')[0], test_raw_psnr, test_raw_ssim) + '\n'
                f1.write(context)
                frame_avg_raw_psnr += test_raw_psnr
                frame_avg_raw_ssim += test_raw_ssim

            output = output*(2**12-1-240)+240
            denoised_raw_frame = preprocess(np.expand_dims(pack_gbrg_raw(output),axis=0))
            denoised_srgb_frame = postprocess(isp(denoised_raw_frame))[0]

            if args.rgb_psnr:
                ### sRGB PSNR
                gt_srgb_frame = cv2.imread(args.input_dir + '/gt/{}/{}'.format(example, imagesNames[int(numOfFrames/2)].split('.')[0]+'.png'), 1)[36:-36].astype(np.float32)/255

                test_srgb_psnr = compare_psnr(gt_srgb_frame, denoised_srgb_frame, data_range=1.0)
                test_srgb_ssim = compare_ssim(gt_srgb_frame, denoised_srgb_frame, data_range=1.0, multichannel=True)
                print('scene {} frame{} test srgb psnr : {}, test srgb ssim : {} '.format(example, imagesNames[int(numOfFrames/2)].split('.')[0], test_srgb_psnr, test_srgb_ssim))
                context = 'scene {} frame{} test srgb psnr : {}, test srgb ssim : {} '.format(example, imagesNames[int(numOfFrames/2)].split('.')[0], test_srgb_psnr, test_srgb_ssim) + '\n'
                f2.write(context)
                frame_avg_srgb_psnr += test_srgb_psnr
                frame_avg_srgb_ssim += test_srgb_ssim

            # Write RGB denoised frame
            denoised_srgb_frame = np.uint8(denoised_srgb_frame*255)
            cv2.imwrite(os.path.join(output_dir, imagesNames[int(numOfFrames/2)].split('.')[0] + '.png'), denoised_srgb_frame)
            counter += 1
    
    ### Save PSNRs to file.
    if args.raw_psnr:
        frame_avg_raw_psnr = frame_avg_raw_psnr/counter
        frame_avg_raw_ssim = frame_avg_raw_ssim/counter
        context = 'frame average raw psnr:{},frame average raw ssim:{} count: {}'.format(frame_avg_raw_psnr,frame_avg_raw_ssim, counter) + '\n'
        f1.write(context)
        print(context[:-1])
    if args.rgb_psnr:
        frame_avg_srgb_psnr = frame_avg_srgb_psnr/counter
        frame_avg_srgb_ssim = frame_avg_srgb_ssim/counter
        context = 'frame average srgb psnr:{},frame average srgb ssim:{} count: {}'.format(frame_avg_srgb_psnr,frame_avg_srgb_ssim, counter) + '\n'
        f2.write(context)
        print(context[:-1])

    return frame_avg_raw_psnr, frame_avg_raw_ssim, frame_avg_srgb_psnr, frame_avg_srgb_ssim


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--input_dir', type=str, default='/data/avinash/rawn/', help='input path')
    parser.add_argument('--output_dir', type=str, default='test/syn/', help='input path')
    parser.add_argument('--model', dest='model', type=str, default='final', help='model type')
    parser.add_argument('--raw_psnr', action='store_true')
    parser.add_argument('--rgb_psnr', action='store_true')
    args = parser.parse_args()

    data_dir = args.input_dir

    examples = os.listdir(data_dir+"iso1600")

    ckpt = args.ckpt
    output_dir = args.output_dir + ckpt + "/"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    denoiser = denoising_raw.DenoiseNet(inp_chans=140).eval()
    denoiser.to(device)
    for param in denoiser.parameters():
        param.requires_grad = False
    
    dict1 = torch.load("model/" + args.model + ".ckpt")
    denoiser.load_state_dict(dict1['state_dict'])

    flowComp = RAFT(args)
    flowComp = flowComp.to(device).eval()
    flowComp.to(device)
    flowComp.load_state_dict(torch.load('core/407000_raft.pth'), strict=False)
    for param in flowComp.parameters():
        param.requires_grad = False
    isp = torch.load('isp/ISP_CNN.pth').to(device)
    for k,v in isp.named_parameters():
        v.requires_grad=False

    backWarp = dataloader.backWarp((1920, 1152), device)
    backWarp = backWarp.to(device)    
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for iso in iso_list:
        context = 'ISO{}'.format(iso) + '\n'
        if args.raw_psnr:
            f1 = open(args.output_dir+'{}_model_test_psnr_and_ssim_on_iso{}_raw.txt'.format(args.ckpt, iso), 'w')
            f1.write(context)
            scene_avg_raw_psnr = 0
            scene_avg_raw_ssim = 0
        if args.rgb_psnr:
            f2 = open(args.output_dir+'{}_model_test_psnr_and_ssim_on_iso{}_sRGB.txt'.format(args.ckpt, iso), 'w')
            f2.write(context)
            scene_avg_srgb_psnr = 0
            scene_avg_srgb_ssim = 0


        for example in examples:
            context = 'scene{}'.format(example) + '\n'

            frame_avg_raw_psnr, frame_avg_raw_ssim, frame_avg_srgb_psnr, frame_avg_srgb_ssim = \
                denoise_syn(os.path.join(data_dir, 'iso' + iso, example), os.path.join(output_dir, 'iso' + iso, example), example)
            
            
            if args.raw_psnr:
                f1.write(context)
                scene_avg_raw_psnr += frame_avg_raw_psnr
                scene_avg_raw_ssim += frame_avg_raw_ssim
            if args.rgb_psnr:
                f2.write(context)
                scene_avg_srgb_psnr += frame_avg_srgb_psnr
                scene_avg_srgb_ssim += frame_avg_srgb_ssim
            
        
        num = len(examples)
        if args.raw_psnr:
            scene_avg_raw_psnr = scene_avg_raw_psnr/num
            scene_avg_raw_ssim = scene_avg_raw_ssim/num
            context = 'scene average raw psnr:{:.4},scene frame average raw ssim:{:.3} #scenes: {}'.format(scene_avg_raw_psnr,scene_avg_raw_ssim, num) + '\n'
            f1.write(context)
            print(context[:-1])
        if args.rgb_psnr:
            scene_avg_srgb_psnr = scene_avg_srgb_psnr/num
            scene_avg_srgb_ssim = scene_avg_srgb_ssim/num
            context = 'scene average srgb psnr:{:.4},scene frame average srgb ssim:{:.3} #scenes: {}'.format(scene_avg_srgb_psnr,scene_avg_srgb_ssim, num) + '\n'
            f2.write(context)
            print(context[:-1])
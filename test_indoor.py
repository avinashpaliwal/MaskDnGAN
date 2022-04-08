from __future__ import division
import os, scipy.io
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import cv2
import argparse
from PIL import Image
from utils import *
from core.raft import RAFT
import denoising_raw
import dataloader
from tqdm import tqdm

# Padding
padder = nn.ZeroPad2d((0, 0, 18, 18))

def forward(input):
    input_raw_left  = input[:,   :4]
    input_raw_centr = input[:,  4:8]
    input_raw_right = input[:, 8:12]

    with torch.no_grad():
        # Calculate optical flows - RAFT
        flow_left, context_centr = flowComp(input_raw_centr, input_raw_left, 32)
        flow_right, _ = flowComp(input_raw_centr, input_raw_right, 32)
        input_raw_left  = trainBackWarp(input_raw_left, flow_left[-1])
        input_raw_right = trainBackWarp(input_raw_right, flow_right[-1])

        # Denoise
        input_raw = torch.cat((input_raw_left, input_raw_centr, input_raw_right, context_centr), 1)
        out_raw = input_raw_centr + denoiser(input_raw)
    
    return out_raw

def denoise(input, prev):
    input = padder(input)
    images = input
    temp = []
    for i in range(3):
        temp.append((prev[-1][i+1]) if (prev[-1] is not None and (i+1) < 3) else forward(images[:, 4*i:4*(i+3)]))
    prev[-1] = temp
    out_raw = forward(torch.cat((prev[-1]), 1))

    return out_raw.detach().cpu().permute(0, 2, 3, 1).numpy()[:, 18:-18], prev

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--model', dest='model', type=str, default='final', help='model type')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--output_dir', type=str, default='./results/indoor/', help='output path')
parser.add_argument('--vis_data', type=bool, default=True, help='whether to visualize noisy and gt data')
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

isp = torch.load('isp/ISP_CNN.pth').cuda()

args.output_dir = args.output_dir + args.model + "/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flowComp = RAFT(args)
flowComp = flowComp.to(device).eval()
flowComp.load_state_dict(torch.load('core/407000_raft.pth'), strict=False)
for param in flowComp.parameters():
    param.requires_grad = False
denoiser = denoising_raw.DenoiseNet(inp_chans=140).eval()
denoiser.to(device)
dict1 = torch.load("model/" + args.model + ".ckpt")
denoiser.load_state_dict(dict1['state_dict'])

trainBackWarp      = dataloader.backWarp((1920, 1152), device)
trainBackWarp      = trainBackWarp.to(device)

iso_list = [25600,12800,6400,3200,1600]

for iso in iso_list:
    print('processing iso={}'.format(iso))

    if not os.path.isdir(args.output_dir+'ISO{}'.format(iso)):
        os.makedirs(args.output_dir+'ISO{}'.format(iso))

    f1 = open(args.output_dir + '{}_model_test_psnr_and_ssim_on_iso{}_raw.txt'.format(args.model, iso), 'w')
    f2 = open(args.output_dir + '{}_model_test_psnr_and_ssim_on_iso{}_sRGB.txt'.format(args.model, iso), 'w')

    context = 'ISO{}'.format(iso) + '\n'
    f1.write(context)
    f2.write(context)
  
    scene_avg_raw_psnr = 0
    scene_avg_raw_ssim = 0
    scene_avg_srgb_psnr = 0
    scene_avg_srgb_ssim = 0

    for scene_id in range(7,11+1):

        context = 'scene{}'.format(scene_id) + '\n'
        f1.write(context)
        f2.write(context)

        frame_avg_raw_psnr = 0
        frame_avg_raw_ssim = 0
        frame_avg_srgb_psnr = 0
        frame_avg_srgb_ssim = 0

        # prev = (None, None)
        prev = [None]

        for i in range(3,5+1):
            frame_list = []
            for j in range(-2,3):
                if (i+j)<1:
                    raw = cv2.imread('../CRVD/CRVD_data/indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame1_noisy0.tiff'.format(scene_id, scene_id, iso),-1)
                    input_full = np.expand_dims(pack_gbrg_raw(raw), axis=0)
                    frame_list.append(input_full)
                elif (i+j)>7:
                    raw = cv2.imread('../CRVD/CRVD_data/indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame7_noisy0.tiff'.format(scene_id, scene_id, iso),-1)
                    input_full = np.expand_dims(pack_gbrg_raw(raw), axis=0)
                    frame_list.append(input_full)
                else:
                    raw = cv2.imread('../CRVD/CRVD_data/indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy0.tiff'.format(scene_id, scene_id, iso, i+j),-1)
                    input_full = np.expand_dims(pack_gbrg_raw(raw), axis=0)
                    frame_list.append(input_full)
            input_data = np.concatenate(frame_list, axis=3)
            
            test_result, prev = denoise(torch.from_numpy(input_data).cuda().permute(0, 3, 1 ,2), prev)
            test_result = depack_gbrg_raw(test_result)

            test_gt = cv2.imread('../CRVD/CRVD_data/indoor_raw_gt/indoor_raw_gt_scene{}/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff'.format(scene_id, scene_id, iso, i),-1).astype(np.float32)
            test_gt = (test_gt-240)/(2**12-1-240)
    
            test_raw_psnr = compare_psnr(test_gt,(np.uint16(test_result*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240), data_range=1.0)
            test_raw_ssim = compute_ssim_for_packed_raw(test_gt, (np.uint16(test_result*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240))
            print('scene {} frame{} test raw psnr : {}, test raw ssim : {} '.format(scene_id, i, test_raw_psnr, test_raw_ssim))
            context = 'scene {} frame{} test raw psnr : {}, test raw ssim : {} '.format(scene_id, i, test_raw_psnr, test_raw_ssim) + '\n'
            f1.write(context)
            frame_avg_raw_psnr += test_raw_psnr
            frame_avg_raw_ssim += test_raw_ssim
                  
            output = test_result*(2**12-1-240)+240
            # save_result = Image.fromarray(np.uint16(output))
            # save_result.save(args.output_dir+'ISO{}/scene{}_frame{}_denoised_raw.tiff'.format(iso, scene_id, i))

            # noisy_raw_frame = preprocess(input_data[:,:,:,4:8])
            # noisy_srgb_frame = postprocess(isp(noisy_raw_frame))[0]
            # if args.vis_data:
            #     cv2.imwrite(args.output_dir+'ISO{}/scene{}_frame{}_noisy_sRGB.png'.format(iso, scene_id, i), np.uint8(noisy_srgb_frame*255))

            denoised_raw_frame = preprocess(np.expand_dims(pack_gbrg_raw(output),axis=0))
            denoised_srgb_frame = postprocess(isp(denoised_raw_frame))[0]
            cv2.imwrite(args.output_dir+'ISO{}/scene{}_frame{}_denoised_sRGB.png'.format(iso, scene_id, i), np.uint8(denoised_srgb_frame*255))

            gt_raw_frame = np.expand_dims(pack_gbrg_raw(test_gt*(2**12-1-240)+240), axis=0)
            gt_srgb_frame = postprocess(isp(preprocess(gt_raw_frame)))[0]
            # if args.vis_data:
            #     cv2.imwrite(args.output_dir+'ISO{}/scene{}_frame{}_gt_sRGB.png'.format(iso, scene_id, i), np.uint8(gt_srgb_frame*255))

            test_srgb_psnr = compare_psnr(np.uint8(gt_srgb_frame*255).astype(np.float32)/255, np.uint8(denoised_srgb_frame*255).astype(np.float32)/255, data_range=1.0)
            test_srgb_ssim = compare_ssim(np.uint8(gt_srgb_frame*255).astype(np.float32)/255, np.uint8(denoised_srgb_frame*255).astype(np.float32)/255, data_range=1.0, multichannel=True)
            print('scene {} frame{} test srgb psnr : {}, test srgb ssim : {} '.format(scene_id, i, test_srgb_psnr, test_srgb_ssim))
            context = 'scene {} frame{} test srgb psnr : {}, test srgb ssim : {} '.format(scene_id, i, test_srgb_psnr, test_srgb_ssim) + '\n'
            f2.write(context)
            frame_avg_srgb_psnr += test_srgb_psnr
            frame_avg_srgb_ssim += test_srgb_ssim

        frame_avg_raw_psnr = frame_avg_raw_psnr/3
        frame_avg_raw_ssim = frame_avg_raw_ssim/3
        frame_avg_srgb_psnr = frame_avg_srgb_psnr/3
        frame_avg_srgb_ssim = frame_avg_srgb_ssim/3
        context = 'frame average raw psnr:{},frame average raw ssim:{}'.format(frame_avg_raw_psnr,frame_avg_raw_ssim) + '\n'
        f1.write(context)
        context = 'frame average srgb psnr:{},frame average srgb ssim:{}'.format(frame_avg_srgb_psnr,frame_avg_srgb_ssim) + '\n'
        f2.write(context)

        scene_avg_raw_psnr += frame_avg_raw_psnr
        scene_avg_raw_ssim += frame_avg_raw_ssim
        scene_avg_srgb_psnr += frame_avg_srgb_psnr
        scene_avg_srgb_ssim += frame_avg_srgb_ssim

    scene_avg_raw_psnr = scene_avg_raw_psnr/5
    scene_avg_raw_ssim = scene_avg_raw_ssim/5
    scene_avg_srgb_psnr = scene_avg_srgb_psnr/5
    scene_avg_srgb_ssim = scene_avg_srgb_ssim/5
    context = 'scene average raw psnr:{},scene frame average raw ssim:{}'.format(scene_avg_raw_psnr,scene_avg_raw_ssim) + '\n'
    f1.write(context)
    context = 'scene average srgb psnr:{},scene frame average srgb ssim:{}'.format(scene_avg_srgb_psnr,scene_avg_srgb_ssim) + '\n'
    f2.write(context)
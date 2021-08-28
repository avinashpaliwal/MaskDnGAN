import torch
import torch.nn as nn

import os
import numpy as np
import random

from core.raft import RAFT
import networks

class GAN_Denoiser(nn.Module):
    def __init__(self, args):
        super(GAN_Denoiser, self).__init__()
        # Save args
        self.args = args

        # Init start epoch
        self.start_epoch = -1
        self.FloatTensor = torch.cuda.FloatTensor

        # Set seed for pytorch, numpy
        self.set_random_seed(args.seed)

        torch.backends.cudnn.benchmark = True

        # Initialize networks
        self.flowComp, self.generator, self.critic, self.ISP = \
            self.initialize_networks()
        self.restore_checkpoints()

        # Initialize loss functions
        self.lossFn = self.initialize_loss_functions()

        # Create checkpoint dir
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
    
    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, warp=None, mode=None):
        if mode == 'generator':
            g_loss, generated = \
                self.compute_generator_loss(data, warp)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(data)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                v_loss, v_generated = \
                    self.compute_validation_loss(data, warp)
            return v_loss, v_generated
        else:
            raise ValueError("|mode| is invalid")
    
    def denoise(self, input, backWarp):
        input_raw_left, input_raw_centr, input_raw_right = input

        with torch.no_grad():
            # Calculate optical flows
            flow_left, context = self.flowComp(input_raw_centr, input_raw_left, 12)
            flow_right, _ = self.flowComp(input_raw_centr, input_raw_right, 12)

            input_raw_left  = backWarp(input_raw_left, flow_left[-1])
            input_raw_right = backWarp(input_raw_right, flow_right[-1])

        input_raw = torch.cat((input_raw_left,
                            input_raw_centr,
                            input_raw_right, 
                            context), 1)
        # Calculate optical flow residuals and visibility maps
        out_raw = input_raw_centr + self.generator(input_raw)

        return out_raw

    def raw2rgb(self, raw, metadata):
        args = self.args

        if isinstance(raw, list) or isinstance(raw, tuple):
            rgb = [self.ISP(im) for im in raw]
        else:
            rgb = self.ISP(raw)

        return rgb

    def generate_fake(self, data, backWarp):
        args = self.args
        input_raw, GT_raw, metadata, gan_mask = data

        return_data = {}

        input_raw = input_raw.to(args.device)
        GT_raw = GT_raw.to(args.device)
        return_data['feature_mask'] = gan_mask.to(args.device).unsqueeze(1)

        input_raw = torch.split(input_raw, 4, 1)
        GT_raw = torch.split(GT_raw, 4, 1)[1:4]

        intermediate_raw = \
            [self.denoise(input_raw[c:c+3], backWarp) for c in range(3)]
        return_data['fake_raw'] = self.denoise(intermediate_raw, backWarp)

        return_data['real_raw'] = GT_raw[1]
        return_data['input_rgb'] = self.raw2rgb(input_raw, metadata)
        return_data['real'] = self.raw2rgb(return_data['real_raw'], metadata)
        return_data['fake'] = self.raw2rgb(return_data['fake_raw'], metadata)

        return return_data

    ######### Private methods #########
    def initialize_networks(self):
        args = self.args
        
        flowComp = RAFT(args)
        flowComp = flowComp.to(args.device).eval()
        for param in flowComp.parameters():
            param.requires_grad = False
        isp = torch.load('isp/ISP_CNN.pth').to(args.device)
        for k,v in isp.named_parameters():
            v.requires_grad=False

        generator = networks.define_G(args.denoise_nc).to(args.device)
        critic = networks.define_D().to(args.device)
        critic.init_weights('xavier', 0.02)

        return flowComp, generator, critic, isp
    
    def restore_checkpoints(self):
        args = self.args

        self.flowComp.load_state_dict(torch.load(args.flow_ckpt))
        print("Loading flow checkpoint ====> " + args.flow_ckpt)

        if args.denoise_ckpt:
            self.generator.load_state_dict(torch.load(args.denoise_ckpt)['state_dict'])
            print("Loading denoise checkpoint ====> " + args.denoise_ckpt)
            self.critic.load_state_dict(torch.load(args.critic_ckpt)['state_dict'])
            print("Loading critic checkpoint ====> " + args.critic_ckpt)
            self.start_epoch = int(args.denoise_ckpt.split('_')[-2])

        return
    
    def create_optimizers(self):
        args = self.args

        G_params = list(self.generator.parameters())
        D_params = list(self.critic.parameters())

        beta1, beta2 = args.beta1, args.beta2
        G_lr, D_lr = args.lr / 2, args.lr * 2

        print("Generator LR: {} Discriminator LR: {}".format(G_lr, D_lr))

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def initialize_loss_functions(self):
        args = self.args
        lossFn = {}

        lossFn['L1']  = nn.L1Loss()
        lossFn['MSE'] = nn.MSELoss()
        lossFn['GAN'] = networks.GANLoss().to(args.device)
        if not args.no_vgg_loss:
            lossFn['VGG'] = networks.VGGLoss().to(args.device)

        return lossFn


    def compute_generator_loss(self, data, warp):
        args = self.args
        G_losses = {}

        generated = self.generate_fake(data, warp)
        feature_mask, fake_image, real_image = \
            generated['feature_mask'], generated['fake'], generated['real']

        pred_fake, pred_real = self.discriminate(feature_mask,
                                                 fake_image,
                                                 real_image)

        G_losses['Reconstruction'] = self.recnLoss(self.lossFn['L1'],
                                                   generated) \
                                                       * args.lambda_recn

        G_losses['GAN'] = self.lossFn['GAN'](pred_fake, True,
                                             for_discriminator=False) \
                                                 * args.lambda_gan

        if not args.no_ganFeat_loss:
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.lossFn['L1'](
                    pred_fake[j], pred_real[j].detach())
                GAN_Feat_loss += unweighted_loss * args.lambda_feat
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not args.no_vgg_loss:
            G_losses['VGG'] = self.lossFn['VGG'](fake_image, real_image) \
                * args.lambda_vgg

        return G_losses, generated
    
    def compute_validation_loss(self, data, warp):
        args = self.args
        G_losses = {}

        generated = self.generate_fake(data, warp)

        G_losses['Reconstruction'] = self.recnLoss(self.lossFn['L1'],
                                                   generated) \
                                                       * args.lambda_recn

        return G_losses, generated

    def compute_discriminator_loss(self, data):
        D_losses = {}

        pred_fake, pred_real = self.discriminate(
            data['feature_mask'], data['fake'].detach(), data['real'].detach())

        D_losses['D_Fake'] = self.lossFn['GAN'](pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_Real'] = self.lossFn['GAN'](pred_real, True,
                                               for_discriminator=True)

        return D_losses

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, feature_mask, fake_image, real_image):
        fake_concat = torch.cat([feature_mask, fake_image], dim=1)
        real_concat = torch.cat([feature_mask, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.critic(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = [tensor[:tensor.size(0) // 2] for tensor in pred]
            real = [tensor[tensor.size(0) // 2:] for tensor in pred]
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real
    
    def recnLoss(self, L1_lossFn, generated):
        args = self.args

        out_rgb, out_raw = generated['fake'], generated['fake_raw']

        # intermediate_left_rgb, intermediate_centr_rgb, intermediate_right_rgb = generated['intermediate_rgb']
        GT_rgb, GT_raw = generated['real'], generated['real_raw']

        # Reconstruction Loss RGB
        recnLoss_rgb = L1_lossFn(out_rgb, GT_rgb)

        # Reconstruction Loss RAW
        recnLoss_raw = L1_lossFn(out_raw, GT_raw)

        loss = recnLoss_raw + 0.5 * (recnLoss_rgb)
        return recnLoss_rgb if (args.rgb) else loss

    def save(self, epoch):
        args = self.args

        save_dict_G = {'state_dict': self.generator.state_dict(), 'args':args}
        save_dict_D = {'state_dict': self.critic.state_dict(), 'args':args}

        torch.save(save_dict_G,
                   os.path.join(args.checkpoint_dir, "{}_ISP_{}_1G.ckpt".format(args.experiment, epoch)))
        torch.save(save_dict_D,
                   os.path.join(args.checkpoint_dir, "{}_ISP_{}_1D.ckpt".format(args.experiment, epoch)))
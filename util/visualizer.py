import torchvision

from tensorboardX import SummaryWriter

class Visualizer():
    # Tensorboard Summary and terminal log
    def __init__(self, args):
        # Initialize tensorboard log writer
        self.writer = SummaryWriter('logs/' + args.experiment)

    def tensorboard_prepare_summary(self, data):
        feature_mask = data['feature_mask'].expand(-1, 3, -1, -1)
        summary = {}

        summary['comp'] = torchvision.utils.make_grid(
            [data['input_rgb'][2].cpu()[0],
             data['real'].cpu()[0],
             data['fake'].cpu()[0],
             feature_mask.cpu()[0]], padding=10)

        summary['noisy'] = torchvision.utils.make_grid(
            [x.cpu()[0] for x in data['input_rgb']], padding=10)
        
        return summary

    def tensorboard_log_summary(self, train_summary, val_summary, itr):
        # Scalars
        self.writer.add_scalars('Loss/train_GAN', 
                                {'train_GAN': train_summary['GAN']}, itr)
        self.writer.add_scalars('Loss/train_Fake', 
                                {'train_Fake': train_summary['D_Fake']}, itr)
        self.writer.add_scalars('Loss/train_Real', 
                                {'train_Real': train_summary['D_Real']}, itr)
        self.writer.add_scalars('Loss/train_Reconstruction', 
                                {'train_Reconstruction': train_summary['Reconstruction']}, itr)
        self.writer.add_scalars('Loss/train_GAN_Feat', 
                                {'train_GAN_Feat': train_summary['GAN_Feat']}, itr)
        self.writer.add_scalars('Loss/train_VGG', 
                                {'train_VGG': train_summary['VGG']}, itr)
        self.writer.add_scalars('Loss/validationLoss', 
                                {'validationLoss': val_summary['loss']}, itr)

        self.writer.add_scalars('PSNR', {'psnr': val_summary['psnr']}, itr)
        
        # Images
        self.writer.add_image('Validation Output', val_summary['comp'], itr)
        self.writer.add_image('Validation Input/Noisy', val_summary['noisy'], itr)
        
        self.writer.add_image('Train Output', train_summary['comp'], itr)
        self.writer.add_image('Train Input/Noisy', train_summary['noisy'], itr)

        # print update
        print(" Iterations: %4d  GAN: %0.6f  Fake: %0.6f  Real: %0.6f  GAN_Feat: %0.6f  VGG: %0.6f  Reconstruction: %0.6f  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f" \
            % (itr, train_summary['GAN'], train_summary['D_Fake'], train_summary['D_Real'], train_summary['GAN_Feat'], train_summary['VGG'], train_summary['Reconstruction'], train_summary['exec_time'], val_summary['loss'], val_summary['psnr'], val_summary['exec_time']))

        return
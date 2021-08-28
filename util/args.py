import argparse

class TrainingArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # general
        self.parser.add_argument("--dir", type=str, required=True, help='path to dataset folder containing train-test-validation folders')
        self.parser.add_argument("--experiment", type=str, default="RAFT_RRG_Grad_Mask_Hinge", help='name of experiment')
        self.parser.add_argument("--num_epochs", type=int, default=200, help='number of epochs to train.')
        self.parser.add_argument("--device", type=str, default="cuda:0", help="set gpu device")
        self.parser.add_argument('--val_freq', type=int, default=800, help='frequency of showing results on screen')
        self.parser.add_argument('--denoise_nc', type=int, default=140, help='frequency of showing results on screen')
        
        # checkpoints
        self.parser.add_argument("--checkpoint_dir", type=str, default="Checkpoints", help='path to folder for saving checkpoints')
        self.parser.add_argument("--flow_ckpt", type=str, default="core/407000_raft.pth", help='path to flow checkpoint')
        self.parser.add_argument("--denoise_ckpt", type=str, help='path to denoise checkpoint')
        self.parser.add_argument("--critic_ckpt", type=str, help='path to discriminator checkpoint')
        
        # for training
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--seed', type=int, default=0, help='random seed value')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lambda_feat', type=float, default=0.01, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=0.005, help='weight for vgg loss')
        self.parser.add_argument('--lambda_gan', type=float, default=0.00005, help='weight for gan loss')
        self.parser.add_argument('--lambda_recn', type=float, default=0.1, help='weight for recn loss')

        # dataset
        self.parser.add_argument('--seq_len', type=int, default=5, help='number of frames in input sequence')
        self.parser.add_argument('--train_patch_size', type=tuple, default=(192, 192), help='training patch size')
        self.parser.add_argument('--val_patch_size', type=tuple, default=(1280, 704), help='validation patch size')
        self.parser.add_argument("--train_batch_size", type=int, default=4, help='batch size for training.')
        self.parser.add_argument("--val_batch_size", type=int, default=2, help='batch size for validation.')
        self.parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataloader")
        self.parser.add_argument('--rgb', action='store_true', help='if specified, use RGB reconstruction loss only')

        # for discriminators
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        
        # RAFT parameters
        self.parser.add_argument('--small', action='store_true', help='use small model')
        self.parser.add_argument('--epsilon', type=float, default=1e-8)
        self.parser.add_argument('--dropout', type=float, default=0.0)
        self.parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
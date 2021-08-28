from torch.utils.data import DataLoader

import os

from data.dataloader import VideoDenoiseDataloader, backWarp

def load_datasets_and_init_backwarp_fn(args):
    loader, backwarp = {}, {}

    trainset = VideoDenoiseDataloader(os.path.join(args.dir, 'train'),
                                      args.seq_len, 
                                      args.train_patch_size)
    loader['train'] = DataLoader(trainset, 
                                 batch_size=args.train_batch_size, 
                                 shuffle=True, 
                                 num_workers=args.num_workers, 
                                 pin_memory=True)

    validationset = VideoDenoiseDataloader(os.path.join(args.dir, 'validation'), 
                                           args.seq_len, 
                                           args.val_patch_size, 
                                           train=False)
    loader['validation'] = DataLoader(validationset, 
                                      batch_size=args.val_batch_size, 
                                      shuffle=True, 
                                      num_workers=args.num_workers, 
                                      pin_memory=True)

    print(trainset, validationset)

    backwarp['train']      = backWarp(args.train_patch_size,
                                      args.device).to(args.device)
    backwarp['validation'] = backWarp(args.val_patch_size,
                                      args.device).to(args.device)

    return loader, backwarp
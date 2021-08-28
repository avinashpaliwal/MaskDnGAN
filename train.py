import torch

import time
from math import log10

from trainer import Trainer
from util.visualizer import Visualizer
from util.args import TrainingArguments
import data

"""### Validation function"""

def validate():
    # For details see training.
    psnr = 0
    vloss = 0
    flag = True
    summary = {}
    start = time.time()

    with torch.no_grad():
        for validationIndex, (validationData) in enumerate(loader['validation'], 0):
            
            loss, generated = trainer.run_validation(validationData,
                                                     warp['validation'])
            
            # For tensorboard
            if (flag):
                val_summary = vis.tensorboard_prepare_summary(generated)
                flag = False
            
            vloss += sum(loss.values()).mean().item()
            
            #psnr
            MSE_val = trainer.model.lossFn['MSE'](generated['fake'], generated['real'])
            psnr += (10 * log10(1 / MSE_val.item()))

    summary['psnr'] = psnr / len(loader['validation'])
    summary['loss'] = vloss / len(loader['validation'])
    summary['exec_time'] = time.time() - start

    return {**summary, **val_summary}

if __name__ == '__main__':

    args = TrainingArguments().parser.parse_args()
    print(args)
    trainer = Trainer(args)
    loader, warp = data.load_datasets_and_init_backwarp_fn(args)
    vis = Visualizer(args)

    """### Training"""
    start = time.time()
    trainer.reset_loss()

    # Test
    val_summary = validate()
    print("Validation PSNR: {} Loss: {} Exec Time: {}".format(
        val_summary['psnr'], val_summary['loss'], val_summary['exec_time']))

    ### Main training loop
    for epoch in range(trainer.start_epoch() + 1, args.num_epochs):
        print("Epoch: ", epoch)        
        
        # Append and reset

        for trainIndex, (trainData) in enumerate(loader['train']):

            trainer.run_generator_one_step(trainData, warp['train'])

            trainer.run_discriminator_one_step()

            trainer.append_loss()

            # Validation and progress every `val_freq` iterations
            if ((trainIndex % args.val_freq) == args.val_freq - 1):
                # Tensorboard Images Train
                train_summary = \
                    vis.tensorboard_prepare_summary(trainer.generated)
                
                train_summary['exec_time'] = time.time() - start
                
                trainer.normalize_loss()

                val_summary = validate()
                
                # Tensorboard
                itr = trainIndex + epoch * (len(loader['train'])) + 1
                vis.tensorboard_log_summary(
                    {**train_summary, **trainer.loss},
                    val_summary, itr)
                
                trainer.reset_loss()
                start = time.time()
        
        # Save checkpoint
        trainer.save(epoch)
from network import GAN_Denoiser

class Trainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """
    def __init__(self, args):
        # Save args
        self.args = args

        self.model = GAN_Denoiser(args)
        
        self.generated = None
        self.loss = None

        # Create optimizers
        self.optimizer_G, self.optimizer_D = \
            self.model.create_optimizers()

    def run_generator_one_step(self, data, warp):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.model(data, warp, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated
    
    def run_validation(self, data, warp):
        return self.model(data, warp, mode='inference')

    def run_discriminator_one_step(self):
        self.optimizer_D.zero_grad()
        d_losses = self.model(self.generated, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def save(self, epoch):
        self.model.save(epoch)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    def start_epoch(self):
        return self.model.start_epoch
    
    def reset_loss(self):
        self.loss = {'Reconstruction': 0,
                     'GAN': 0,
                     'GAN_Feat': 0,
                     'VGG': 0,
                     'D_Fake': 0,
                     'D_Real': 0}
    
    def append_loss(self):
        for (key, value) in self.get_latest_losses().items():
            self.loss[key] += value.item()
    
    def normalize_loss(self):
        for (key, value) in self.loss.items():
            self.loss[key] /= self.args.val_freq
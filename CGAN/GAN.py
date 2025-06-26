from abc import abstractmethod
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn.functional as F
from cbam import CBAM
import cv2
from torchvision.models import resnet18
import yaml
from torch.nn.functional import binary_cross_entropy


class Generator(nn.Module):
    "Assumes that the input images are 64x64"
    def __init__(self, latent_size, lr):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_size=latent_size
        self.model=self.build_generator()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
    
    def gen_loss_function(self, d_synth):
        t_synth=torch.ones_like(d_synth)
        return binary_cross_entropy(d_synth,t_synth)
    
    @abstractmethod
    def forward(self, x, y):
        raise NotImplementedError("Must be implemented in subclass")
 
    @abstractmethod
    def build_generator(self):
        raise NotImplementedError("Must be implemented in subclass")

class Discriminator(nn.Module):
    "Assumes that the input images are 64x64"
    def __init__(self, label_smoothing, lr):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_smoothing = label_smoothing
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
    
    def disc_loss_function(self, d_true,d_synth):
        t_true=torch.ones_like(d_true) - self.label_smoothing
        t_synth=torch.zeros_like(d_synth) + self.label_smoothing
        return binary_cross_entropy(d_true,t_true) + binary_cross_entropy(d_synth,t_synth)
    
    @abstractmethod
    def forward(self, x, y):
        raise NotImplementedError("Must be implemented in subclass")
    
    @abstractmethod
    def build_discriminator(self):
        raise NotImplementedError("Must be implemented in subclass")

class GAN(nn.Module):
    "Assumes that the input images are 64x64"
    def __init__(self, architecture_yaml, train_yaml):
        #latent_size=128, encoder_channel_progression=[32, 64, 128, 128], decoder_channel_progression=[128, 128, 64, 32]):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture_yaml=architecture_yaml
        self.train_yaml=train_yaml
        
        # Load the configuration file
        with open(self.architecture_yaml, "r") as f:
            self.arch_config = yaml.safe_load(f)
        with open(self.train_yaml, "r") as f:
            self.param_config = yaml.safe_load(f)
        
        self.latent_size = self.arch_config["LATENT_SIZE"]
        self.label_smoothing = self.param_config["TRAINING"]["LABEL_SMOOTHING"]
        self.lr_gen = self.param_config["TRAINING"]["GENERATOR"]["LR"]
        self.lr_disc = self.param_config["TRAINING"]["DISCRIMINATOR"]["LR"]
        self.num_epochs = self.param_config["TRAINING"]["NUM_EPOCHS"]
    
    def train(self, checkpoint_path, dataloader):
        self.generator.model.train()
        self.discriminator.model.train()
        disc_optimizer, gen_optimizer = self.discriminator.optimizer, self.generator.optimizer
        list_loss_generator = []
        list_loss_discriminator = []
        list_sum_dtrue = []
        list_sum_dsynth = []
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

            self.generator.model.load_state_dict(checkpoint['model_state_dict_generator'])
            self.discriminator.model.load_state_dict(checkpoint['model_state_dict_discriminator'])
            gen_optimizer.load_state_dict(checkpoint['optimizer_state_dict_generator'])
            disc_optimizer.load_state_dict(checkpoint['optimizer_state_dict_discriminator'])
            start_epoch = checkpoint['epoch']
            list_loss_generator = checkpoint['list_loss_generator']
            list_loss_discriminator = checkpoint['list_loss_discriminator']
            list_sum_dtrue = checkpoint['list_sum_dtrue']
            list_sum_dsynth = checkpoint['list_sum_dsynth']
            print(f"Riprendo l'addestramento da epoca {start_epoch+1}...")
        
        for epoch in range(start_epoch, self.num_epochs, 1):
            loss_generator, loss_discriminator, sum_dtrue, sum_dsynth = self.train_one_epoch(dataloader)
            list_loss_generator.append(loss_generator)
            list_loss_discriminator.append(loss_discriminator)
            list_sum_dtrue.append(sum_dtrue)
            list_sum_dsynth.append(sum_dsynth)
        
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict_generator': self.generator.model.state_dict(),
                'model_state_dict_discriminator': self.discriminator.model.state_dict(),
                'optimizer_state_dict_generator': gen_optimizer.state_dict(),
                'optimizer_state_dict_discriminator': disc_optimizer.state_dict(),
                'list_loss_generator': list_loss_generator,
                'list_loss_discriminator': list_loss_discriminator,
                'list_sum_dtrue': list_sum_dtrue,
                'list_sum_dsynth': list_sum_dsynth
            }
            torch.save(checkpoint, checkpoint_path)
        print("FINISH TRANING")

    def train_one_epoch(self, dataloader):
        sum_gloss = 0.0
        sum_dloss = 0.0
        sum_dtrue = 0.0
        sum_dsynth = 0.0
        batches = 0
        disc_optimizer, gen_optimizer = self.discriminator.optimizer, self.generator.optimizer
        for x_true, cls in dataloader:
            x_true=x_true.to(self.device)
            d_true=self.discriminator.model(x_true, cls)
            z=torch.randn(x_true.shape[0], self.latent_size, device=self.device)
            x_synth=self.generator.model(z, cls)
            d_synth = self.discriminator.model(x_synth, cls)

            # update the discriminator
            disc_optimizer.zero_grad()
            dloss=self.discriminator.disc_loss_function(d_true,d_synth)
            dloss.backward(retain_graph=True)
            disc_optimizer.step()

            #update the generator
            d_synth=self.discriminator.model(x_synth, cls)
            gen_optimizer.zero_grad()
            gloss=self.generator.gen_loss_function(d_synth)
            gloss.backward()
            gen_optimizer.step()

            sum_gloss += gloss.detach().cpu().item()
            sum_dloss += dloss.detach().cpu().item()
            sum_dtrue += d_true.mean().detach().cpu().item()
            sum_dsynth += d_synth.mean().detach().cpu().item()
            batches += 1
        print(f'GLoss: {sum_gloss/batches}, DLoss: {sum_dloss/batches}, DTrue: {sum_dtrue/batches}, DSynth: {sum_dsynth/batches}')
        return sum_gloss/batches, sum_dloss/batches, sum_dtrue/batches, sum_dsynth/batches


class BaselineGAN(GAN):
    def __init__(self, architecture_yaml, train_yaml):
        super().__init__(architecture_yaml, train_yaml)
        self.generator_channel_progression = self.arch_config['GENERATOR_CHANNEL_PROGRESSION']
        self.discriminator_channel_progression = self.arch_config['DISCRIMINATOR_CHANNEL_PROGRESSION']
        self.discriminator = BaselineDiscriminator(self.discriminator_channel_progression)
        

class BaselineDiscriminator(Discriminator):
    def __init__(self, discriminator_channel_progression):
        super().__init__()
        self.discriminator_channel_progression=discriminator_channel_progression
        self.model=self.build_discriminator()

        
    def build_discriminator(self):
        model=nn.Sequential()
        prev=3 + 3      # prev è il numero di feature maps iniziale (nel caso di condizionale ad 8 classi dobbiamo avere 3-RGB + 3-Classi = 6)
        size=64    # size identifica la dimensione dell'immagine di input (celebA è 64)
        for k in self.discriminator_channel_progression:
            model.append(nn.Conv2d(prev, k, 3, padding='same'))
            model.append(nn.LeakyReLU())
            model.append(nn.Conv2d(k, k, 3, stride=2, padding=1))
            model.append(nn.LeakyReLU())
            prev=k
            size=size//2    # al secondo Conv2D viene dimezzata la dimensione di size
        model.append(nn.Flatten())
        features=k*size*size    # numero di features a valle del flatten
        model.append(nn.Linear(features, k))
        model.append(nn.LeakyReLU())
        model.append(nn.Dropout(p=0.3))
        model.append(nn.Linear(k, k//2))
        model.append(nn.LeakyReLU())
        model.append(nn.Dropout(p=0.3))
        model.append(nn.Linear(k//2, 2))
        model.append(nn.Sigmoid())
        
        # assert size==8
        # assert k==128
        return model

    def forward(self, x, c):
        cond = c.unsqueeze(2).unsqueeze(3).expand(-1, -1, 64, 64)
        out = self.model(torch.cat([x, cond], dim=1))
        return out
        
    
class BaselineGenerator(Generator):
    def __init__(self, generator_channel_progression):
        super().__init__()
        self.generator_channel_progression=generator_channel_progression
        self.model=self.build_generator()

    def build_generator(self):
        model=nn.Sequential()
        size=4
        prev=128
        model.append(nn.Linear(self.latent_size + 3, prev*size*size))
        model.append(nn.ReLU())
        model.append(nn.Unflatten(1, (prev, size, size)))
        for k in self.generator_channel_progression:
            model.append(nn.Conv2d(prev, k, 3, padding='same'))
            model.append(nn.ReLU())
            model.append(nn.ConvTranspose2d(k, k, 3, stride=2, padding=1,
                                            output_padding=1))
            model.append(nn.ReLU())
            prev=k
            size=size*2
        # assert size==128
        # assert k==32
        model.append(nn.Conv2d(k, 3, 3, padding='same'))
        model.append(nn.Sigmoid())
        return model
    
    def forward(self, z, c):
        zc = torch.cat([z,c], dim=1)
        return self.model(zc)

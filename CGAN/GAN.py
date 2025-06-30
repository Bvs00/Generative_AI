import os
os.environ["MPLCONFIGDIR"] = os.path.expanduser("/tmp/matplotlib")
from abc import abstractmethod
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from cbam import CBAM
import cv2
from torchvision.models import resnet18
import yaml
from torch.nn.functional import binary_cross_entropy
import sys

class Generator(nn.Module):
    "Assumes that the input images are 64x64"
    def __init__(self, architecture_yaml, train_yaml):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.latent_size = architecture_yaml["GENERATOR"]["LATENT_SIZE"]
        self.lr_gen = train_yaml["TRAINING"]["GENERATOR"]["LR"]

        self.model=self.build_generator()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr_gen)
    
    def gen_loss_function(self, d_synth):
        t_synth=torch.ones_like(d_synth)
        return binary_cross_entropy(d_synth,t_synth)
    
    @abstractmethod
    def forward(self, x, y):
        raise NotImplementedError("Must be implemented in subclass")
 
    @abstractmethod
    def build_generator(self):
        raise NotImplementedError("Must be implemented in subclass")
    
    def generate_sample(self, y, save_folder, time):
        self.eval()
        z = torch.randn(size=(self.latent_size,))
        with torch.no_grad():
            image_generated = self(z.unsqueeze(0).to(self.device),y.unsqueeze(0).to(self.device))
        os.makedirs(save_folder, exist_ok=True)
        img = (image_generated.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_folder, f'generated_{time}.png'), img_bgr)

class Discriminator(nn.Module):
    "Assumes that the input images are 64x64"
    def __init__(self, architecture_yaml, train_yaml):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_smoothing = train_yaml["TRAINING"]["LABEL_SMOOTHING"]
        self.lr_disc = train_yaml["TRAINING"]["DISCRIMINATOR"]["LR"]

        self.model=self.build_discriminator()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr_disc)
    
    def disc_loss_function(self, d_true, d_synth):
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
        
        # Extract parameters from the YAML files
        self.num_epochs = self.train_yaml["TRAINING"]["NUM_EPOCHS"]

        # Load the generator class from the architecture YAML
        self.generator = globals()[architecture_yaml["GENERATOR"]["CLASS_NAME"]](architecture_yaml, train_yaml)
        self.generator.to(self.device)

        # Load the discriminator class from the architecture YAML
        self.discriminator = globals()[architecture_yaml["DISCRIMINATOR"]["CLASS_NAME"]](architecture_yaml, train_yaml)
        self.discriminator.to(self.device)
    
    def training_epoch(self, checkpoint_path, dataloader):
        self.generator.train()
        self.discriminator.train()
        disc_optimizer, gen_optimizer = self.discriminator.optimizer, self.generator.optimizer
        list_loss_generator = []
        list_loss_discriminator = []
        list_sum_dtrue = []
        list_sum_dsynth = []
        start_epoch=0
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

            self.generator.load_state_dict(checkpoint['model_state_dict_generator'])
            self.discriminator.load_state_dict(checkpoint['model_state_dict_discriminator'])
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
            print(f'Epoch: {epoch+1}\n\tGLoss: {loss_generator}, DLoss: {loss_discriminator}\n\t\tDTrue: {sum_dtrue}, DSynth: {sum_dsynth}')
            list_loss_generator.append(loss_generator)
            list_loss_discriminator.append(loss_discriminator)
            list_sum_dtrue.append(sum_dtrue)
            list_sum_dsynth.append(sum_dsynth)
        
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict_generator': self.generator.state_dict(),
                'model_state_dict_discriminator': self.discriminator.state_dict(),
                'optimizer_state_dict_generator': gen_optimizer.state_dict(),
                'optimizer_state_dict_discriminator': disc_optimizer.state_dict(),
                'list_loss_generator': list_loss_generator,
                'list_loss_discriminator': list_loss_discriminator,
                'list_sum_dtrue': list_sum_dtrue,
                'list_sum_dsynth': list_sum_dsynth
            }
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
        print("FINISH TRANING")

    def train_one_epoch(self, dataloader):
        sum_gloss = 0.0
        sum_dloss = 0.0
        sum_dtrue = 0.0
        sum_dsynth = 0.0
        batches = 0
        disc_optimizer, gen_optimizer = self.discriminator.optimizer, self.generator.optimizer
        for x_true, cls in tqdm(dataloader):
        # for x_true, cls in dataloader:
            x_true=x_true.to(self.device)
            cls = cls.to(self.device)
            d_true=self.discriminator(x_true, cls)
            z=torch.randn(x_true.shape[0], self.generator.latent_size, device=self.device)
            x_synth=self.generator(z, cls)
            d_synth = self.discriminator(x_synth, cls)

            # update the discriminator
            disc_optimizer.zero_grad()
            dloss=self.discriminator.disc_loss_function(d_true,d_synth)
            dloss.backward(retain_graph=True)
            disc_optimizer.step()

            #update the generator
            d_synth=self.discriminator(x_synth, cls)
            gen_optimizer.zero_grad()
            gloss=self.generator.gen_loss_function(d_synth)
            gloss.backward()
            gen_optimizer.step()

            sum_gloss += gloss.detach().cpu().item()
            sum_dloss += dloss.detach().cpu().item()
            sum_dtrue += d_true.mean().detach().cpu().item()
            sum_dsynth += d_synth.mean().detach().cpu().item()
            batches += 1
        
        return sum_gloss/batches, sum_dloss/batches, sum_dtrue/batches, sum_dsynth/batches


######################  BASELINE GAN ################################    
class BaselineDiscriminator(Discriminator):
    def __init__(self, architecture_yaml, train_yaml):
        self.discriminator_channel_progression = architecture_yaml["DISCRIMINATOR"]['CHANNEL_PROGRESSION']
        # check if self.initial_channel_dim is already defined (in an inherited class)
        if 'initial_channel_dim' not in self.__dict__:
            print("initial_channel_dim not defined, setting to default value of 6")
            self.initial_channel_dim = 3 + 3  # prev è il numero di feature maps iniziale (nel caso di condizionale ad 8 classi dobbiamo avere 3-RGB + 3-Classi = 6)
        self.wh_size = 64    # size identifica la dimensione dell'immagine di input (celebA è 64)
        super().__init__(architecture_yaml, train_yaml)

    def build_discriminator(self):
        model=nn.Sequential()
        for k in self.discriminator_channel_progression:
            model.append(nn.Conv2d(self.initial_channel_dim, k, 3, padding='same'))
            nn.BatchNorm2d(k),
            model.append(nn.LeakyReLU())
            model.append(nn.Conv2d(k, k, 3, stride=2, padding=1))
            nn.BatchNorm2d(k),
            model.append(nn.LeakyReLU())
            self.initial_channel_dim=k
            self.wh_size=self.wh_size//2    # al secondo Conv2D viene dimezzata la dimensione di self.wh_size
        model.append(nn.Flatten())
        features=k*2*self.wh_size    # numero di features a valle del flatten
        model.append(nn.Linear(features, k))
        model.append(nn.LeakyReLU())
        model.append(nn.Dropout(p=0.3))
        model.append(nn.Linear(k, 1))
        model.append(nn.Sigmoid())
        
        # assert size==8
        # assert k==128
        return model

    def forward(self, x, c):
        cond = c.unsqueeze(2).unsqueeze(3).expand(-1, -1, 64, 64)
        out = self.model(torch.cat([x, cond], dim=1))
        return out
        
    
class BaselineGenerator(Generator):
    def __init__(self, architecture_yaml, train_yaml):
        self.generator_channel_progression=architecture_yaml["GENERATOR"]['CHANNEL_PROGRESSION']
        super().__init__(architecture_yaml, train_yaml)
        self.cond_latent = nn.Embedding(8, self.latent_size//8).to(self.device)  # Embedding per le 8 classi condizionali
        self.powers = 2 ** torch.arange(3 - 1, -1, -1).to(self.device)

    def build_generator(self):
        model=nn.Sequential()
        size=4
        prev=self.generator_channel_progression[0]
        model.append(nn.Linear(self.latent_size + self.latent_size//8, (prev//2)*size*size))
        model.append(nn.GELU())
        model.append(nn.Linear((prev//2)*size*size, prev*size*size))
        model.append(nn.GELU())
        model.append(nn.Unflatten(1, (prev, size, size)))
        for k in self.generator_channel_progression:
            model.append(nn.Conv2d(prev, k, 3, padding='same'))
            model.append(nn.BatchNorm2d(k))
            model.append(nn.GELU())
            model.append(nn.Conv2d(k, k, 3, padding='same'))
            model.append(nn.BatchNorm2d(k))
            model.append(nn.GELU())
            model.append(nn.ConvTranspose2d(k, k, 3, stride=2, padding=1,
                                            output_padding=1))
            model.append(nn.GELU())
            prev=k
            size=size*2
        # assert size==128
        # assert k==32
        model.append(nn.Conv2d(k, 3, 3, padding='same'))
        model.append(nn.Sigmoid())
        return model
    
    def forward(self, z, c):

        # transform the class c into an integer
        integers = (c * self.powers).sum(dim=1).long()
        c_token = self.cond_latent(integers)

        zc = torch.cat([z,c_token], dim=1)
        return self.model(zc)
#########################################################################################################


##########################      RESNET GAN  #############################
class ResNetDiscriminator(Discriminator):
    def __init__(self, label_smoothing, lr_disc):
        # check if self.initial_channel_dim is already defined (in an inherited class)
        if 'initial_channel_dim' not in self.__dict__:
            print("initial_channel_dim not defined, setting to default value of 6")
            self.initial_channel_dim = 3 + 3  # prev è il numero di feature maps iniziale (nel caso di condizionale ad 8 classi dobbiamo avere 3-RGB + 3-Classi = 6)
        
        super().__init__(label_smoothing, lr_disc)
        
    def build_discriminator(self):
        base_model = resnet18()
        base_model.conv1=nn.Conv2d(self.initial_channel_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        return nn.Sequential(
            nn.Sequential(*list(base_model.children())[:-2]),  # Remove avgpool & fc
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        cond = c.unsqueeze(2).unsqueeze(3).expand(-1, -1, 64, 64)
        out = self.model(torch.cat([x, cond], dim=1))
        return out


class ResNetVDiscriminator(ResNetDiscriminator):
    def __init__(self, architecture_yaml, train_yaml):
        self.initial_channel_dim = 3 + 3 + 1     # prev è il numero di feature maps iniziale (nel caso di condizionale ad 8 classi dobbiamo avere 3-RGB + 3-Classi = 6)
        super().__init__(architecture_yaml, train_yaml)
    
    def forward(self, x, c):
        BATCH_SIZE = x.shape[0]
        # compute the variance of x pixel by pixel among the batch dimension
        x_var = torch.var(x, dim=[0, 1], keepdim=True).expand(BATCH_SIZE, -1, -1, -1)  # expand to match the input shape
        # concatenate the variance channel to the input
        x = torch.cat([x, x_var], dim=1)

        out = super().forward(x, c)
        return out
        

class BasicGenerator(Generator):
    def __init__(self, latent_size, lr_gen):
        super().__init__(latent_size, lr_gen)
        self.model=self.build_generator()
        self.fc_generator = nn.Linear(self.latent_size + 3, 512 * 4 * 4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr_gen)

    def build_generator(self):
        return nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 64x64
            nn.Sigmoid()  # output in [0,1]
        )
    
    def forward(self, z, c):
        zc = self.fc_generator(torch.cat([z,c], dim=1)).view(-1, 512, 4, 4)
        return self.model(zc)

    def generate_sample(self, y, save_folder, time):
        self.eval()
        z = torch.randn(size=(self.latent_size,))
        with torch.no_grad():
            image_generated = self.decoder(self.fc_decoder(torch.cat([z.unsqueeze(0).to(self.device),y.unsqueeze(0).to(self.device)], dim=1)).view(-1, 512, 4, 4))
        os.makedirs(save_folder, exist_ok=True)
        img = (image_generated.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_folder, f'generated_{time}.png'), img_bgr)
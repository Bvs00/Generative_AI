from abc import abstractmethod
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn.functional as F

class VAutoEncoder(nn.Module):
    "Assumes that the input images are 64x64"
    def __init__(self, latent_size):
        #latent_size=128, encoder_channel_progression=[32, 64, 128, 128], decoder_channel_progression=[128, 128, 64, 32]):
        super().__init__()
        self.latent_size=latent_size
        self.encoder=self.build_encoder()
        self.decoder=self.build_decoder()
 
    def forward(self, x, y):
        cond = y.unsqueeze(2).unsqueeze(3).expand(-1, -1, 64, 64)
        out=self.encoder(torch.cat([x,cond], dim=1))         # l'outpur dell'encoder è 2*LATENT_SIZE
        mu=out[:, :self.latent_size]    # da 0 a LATENT_SIZE corrisponde a mu
        log_sigma=out[:, self.latent_size:]     # da LATENT_SIZE fino alla fine corrisponde a log_sigma
        eps=torch.randn_like(mu)        # generazione del vettore latente secondo la normale
        z=eps*torch.exp(log_sigma)+mu       # generazione del vettore latente 
        y=self.decoder(torch.cat([z,y], dim=1))
        return y, mu, log_sigma
 
    def encode_mu(self, x):
        out=self.encoder(x)
        mu=out[:, :self.latent_size]
        return mu
 
    @abstractmethod
    def build_encoder(self):
        raise NotImplementedError("Must be implemented in subclass")
    
    @abstractmethod
    def build_decoder(self):
        raise NotImplementedError("Must be implemented in subclass")
    
    def generate_sample(self, y, save_folder, device='cpu'):
        self.eval()
        z = torch.randn(size=(self.latent_size,))
        with torch.no_grad():
            image_generated = self.decoder(torch.cat([z.unsqueeze(0).to(device),y.unsqueeze(0).to(device)], dim=1))
        
        plt.figure()
        plt.imshow(image_generated[0].cpu().permute(1,2,0))     # CxHxW -> HxWxC
        plt.axis('off')
        label_str = '_'.join(str(int(v)) for v in y.tolist())
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, label_str))


    def train_one_epoch(self, criterion, optimizer, train_loader, device):
        average_loss = 0.0
        for x_batch, label_batch in tqdm(train_loader, dynamic_ncols=True):
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)
            
            output, mu, log_sigma = self.forward(x_batch, label_batch)
            
            loss = criterion(output, x_batch, mu, log_sigma)
            
            loss.backward()
            optimizer.step()
            
            average_loss += loss
        return average_loss
    

class BaselineVAE(VAutoEncoder):
    def __init__(self, architecture_yaml):
        self.latent_size = architecture_yaml['LATENT_SIZE']
        self.encoder_channel_progression = architecture_yaml['ENCODER_CHANNEL_PROGRESSION']
        self.decoder_channel_progression = architecture_yaml['DECODER_CHANNEL_PROGRESSION']
        super().__init__(self.latent_size)

    def build_encoder(self):
        model=nn.Sequential()
        prev=3 + 3      # prev è il numero di feature maps iniziale (nel caso di condizionale ad 8 classi dobbiamo avere 3-RGB + 3-Classi = 6)
        size=64    # size identifica la dimensione dell'immagine di input (celebA è 64)
        for k in self.encoder_channel_progression:
            model.append(nn.Conv2d(prev, k, 3, padding='same'))
            model.append(nn.ReLU())
            model.append(nn.Conv2d(k, k, 3, stride=2, padding=1))
            model.append(nn.ReLU())
            prev=k
            size=size//2    # al secondo Conv2D viene dimezzata la dimensione di size
        model.append(nn.Flatten())
        features=k*size*size    # numero di features a valle del flatten
        model.append(nn.Linear(features, 2*self.latent_size))
        # assert size==8
        # assert k==128
        return model
 
    def build_decoder(self):
        model=nn.Sequential()
        size=4
        prev=128
        model.append(nn.Linear(self.latent_size + 3, prev*size*size))
        model.append(nn.ReLU())
        model.append(nn.Unflatten(1, (prev, size, size)))
        for k in self.decoder_channel_progression:
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
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x) if len(self.skip) > 0 else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class Res_VAE(VAutoEncoder):
    def __init__(self, architecture_yaml):

        self.latent_size = architecture_yaml['LATENT_SIZE']
        
        self.img_channels=3 
        self.cond_dim=3
        super().__init__(self.latent_size)

    def build_encoder(self):
        return nn.Sequential(
            nn.Conv2d(self.img_channels, 64, kernel_size=3, stride=2, padding=1),  # 64x32x32
            nn.ReLU(),
            ResidualBlock(64, 128, stride=2),  # 128x16x16
            ResidualBlock(128, 256, stride=2),  # 256x8x8
            ResidualBlock(256, 512, stride=2),  # 512x4x4
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 2*self.latent_size)
        )

    def build_decoder(self):
        model = nn.Sequential()
        size = 4
        prev = 256

        # Fully connected and reshape
        model.append(nn.Linear(self.latent_size + 3, prev * size * size))
        model.append(nn.ReLU())
        model.append(nn.Unflatten(1, (prev, size, size)))

        channels = [256, 128, 64, 32]  # Progressive decoding
        for k in channels:
            model.append(ResidualBlock(prev, prev))
            model.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            model.append(nn.Conv2d(prev, k, kernel_size=3, padding=1))
            model.append(nn.BatchNorm2d(k))
            model.append(nn.ReLU())
            prev = k
            size *= 2

        # Final convolution to image
        model.append(nn.Conv2d(prev, 3, kernel_size=3, padding=1))
        model.append(nn.Sigmoid())  # or Tanh if you normalize images to [-1, 1]

        return model

    def forward(self, x, y):
        # skip_connections = []
        h = x
        for layer in self.encoder[:-3]:
            h = layer(h)
            # if isinstance(layer, ResidualBlock):
            #     skip_connections.append(h)

        for layer in self.encoder[-3:]:
            h = layer(h)

        mu=h[:, :self.latent_size]    # da 0 a LATENT_SIZE corrisponde a mu
        log_sigma=h[:, self.latent_size:]     # da LATENT_SIZE fino alla fine corrisponde a log_sigma
        eps=torch.randn_like(mu)        # generazione del vettore latente secondo la normale
        z=eps*torch.exp(log_sigma)+mu       # generazione del vettore latente
        out=torch.cat([z,y], dim=1)

        # skip_idx = -1
        # for i, layer in enumerate(self.decoder):
            # if isinstance(layer, nn.Upsample):
            #     out = layer(out)
            #     if skip_idx >= -len(skip_connections):
            #         skip = skip_connections[skip_idx]
            #         skip = F.interpolate(skip, size=out.shape[2:])
            #         out = torch.cat([out, skip], dim=1)
            #         skip_idx -= 1
            # else:
                # out = layer(out)
        out = self.decoder(out)

        return out, mu, log_sigma

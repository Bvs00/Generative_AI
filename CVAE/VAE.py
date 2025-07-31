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
import numpy as np

######### LOSS ############
class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super(VAELoss, self).__init__()
        self.beta = beta
        self.reconstruction_loss_function = nn.BCELoss(reduction='sum')

    def kl_loss_function(self, mu, log_sigma):
        log_sigma_2 = 2*log_sigma
        kl = 0.5*(mu**2+torch.exp(log_sigma_2)-1-log_sigma_2)
        return torch.sum(kl)

    def forward(self, reconstructed, original, mu, log_sigma):
        return self.reconstruction_loss_function(reconstructed, original) + \
            self.beta*self.kl_loss_function(mu, log_sigma)

class VAutoEncoder(nn.Module):
    "Assumes that the input images are 64x64"
    def __init__(self, architecture_yaml, info_yaml):
        #latent_size=128, encoder_channel_progression=[32, 64, 128, 128], decoder_channel_progression=[128, 128, 64, 32]):
        super().__init__()

        self.num_epochs = info_yaml["TRAINING"]["NUM_EPOCHS"]
        self.latent_size = architecture_yaml["LATENT_SIZE"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder=self.build_encoder()
        self.decoder=self.build_decoder()

        self.criterion = VAELoss(beta=info_yaml["TRAINING"]["BETA"])
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=info_yaml["TRAINING"]["LR"])
 
    @abstractmethod
    def forward(self, x, y):
        raise NotImplementedError("Must be implemented in subclass")
 
    @abstractmethod
    def build_encoder(self):
        raise NotImplementedError("Must be implemented in subclass")
    
    @abstractmethod
    def build_decoder(self):
        raise NotImplementedError("Must be implemented in subclass")
    
    def generate_sample(self, y, save_folder, time):
        self.eval()
        z = torch.randn(size=(self.latent_size,))
        with torch.no_grad():
            image_generated = self.decoder(torch.cat([z.unsqueeze(0).to(self.device),y.unsqueeze(0).to(self.device)], dim=1))
        
        os.makedirs(save_folder, exist_ok=True)

        img = (image_generated.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_folder, f'generated_{time}.png'), img_bgr)
        
    def train_one_epoch(self, train_loader):
        average_loss = 0.0
        for x_batch, label_batch in tqdm(train_loader, dynamic_ncols=True):
        # for x_batch, label_batch in train_loader:
            self.optimizer.zero_grad()
            x_batch = x_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            
            output, mu, log_sigma = self.forward(x_batch, label_batch)
            
            loss = self.criterion(output, x_batch, mu, log_sigma)
            
            loss.backward()
            self.optimizer.step()
            
            average_loss += loss.detach()
        return average_loss

    def training_epoch(self, checkpoint_path, dataloader):
        start_epoch = 0
        list_loss_train = []
        self.train()

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            list_loss_train = checkpoint['list_loss_train']
            print(f"Riprendo l'addestramento da epoca {start_epoch+1}...")
        
        for epoch in range(start_epoch, self.num_epochs, 1):
            average_loss = self.train_one_epoch(dataloader)

            print(f"Epoch {epoch+1} completed. Average loss = {average_loss/(len(dataloader)*1000)}")
            list_loss_train.append(average_loss/len(dataloader))
            
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'list_loss_train': list_loss_train,
            }
            torch.save(checkpoint, checkpoint_path)

        print("FINISH TRANING")
    

# class BaselineVAE(VAutoEncoder):
#     def __init__(self, architecture_yaml, info_yaml):        
#         self.latent_size = architecture_yaml['LATENT_SIZE']
#         self.encoder_channel_progression = architecture_yaml['ENCODER_CHANNEL_PROGRESSION']
#         self.decoder_channel_progression = architecture_yaml['DECODER_CHANNEL_PROGRESSION']
#         super().__init__(architecture_yaml, info_yaml)

#     def build_encoder(self):
#         model=nn.Sequential()
#         prev=3 + 3      # prev è il numero di feature maps iniziale (nel caso di condizionale ad 8 classi dobbiamo avere 3-RGB + 3-Classi = 6)
#         size=64    # size identifica la dimensione dell'immagine di input (celebA è 64)
#         for k in self.encoder_channel_progression:
#             model.append(nn.Conv2d(prev, k, 3, padding='same'))
#             model.append(nn.ReLU())
#             model.append(nn.Conv2d(k, k, 3, stride=2, padding=1))
#             model.append(nn.ReLU())
#             prev=k
#             size=size//2    # al secondo Conv2D viene dimezzata la dimensione di size
#         model.append(nn.Flatten())
#         features=k*size*size    # numero di features a valle del flatten
#         model.append(nn.Linear(features, 2*self.latent_size))
#         # assert size==8
#         # assert k==128
#         return model
 
#     def build_decoder(self):
#         model=nn.Sequential()
#         size=4
#         prev=128
#         model.append(nn.Linear(self.latent_size + 3, prev*size*size))
#         model.append(nn.ReLU())
#         model.append(nn.Unflatten(1, (prev, size, size)))
#         for k in self.decoder_channel_progression:
#             model.append(nn.Conv2d(prev, k, 3, padding='same'))
#             model.append(nn.ReLU())
#             model.append(nn.ConvTranspose2d(k, k, 3, stride=2, padding=1,
#                                             output_padding=1))
#             model.append(nn.ReLU())
#             prev=k
#             size=size*2
#         # assert size==128
#         # assert k==32
#         model.append(nn.Conv2d(k, 3, 3, padding='same'))
#         model.append(nn.Sigmoid())
#         return model

#     def forward(self, x, y):
#         cond = y.unsqueeze(2).unsqueeze(3).expand(-1, -1, 64, 64)
#         out=self.encoder(torch.cat([x,cond], dim=1))         # l'outpur dell'encoder è 2*LATENT_SIZE
#         mu=out[:, :self.latent_size]    # da 0 a LATENT_SIZE corrisponde a mu
#         log_sigma=out[:, self.latent_size:]     # da LATENT_SIZE fino alla fine corrisponde a log_sigma
#         eps=torch.randn_like(mu)        # generazione del vettore latente secondo la normale
#         z=eps*torch.exp(log_sigma)+mu       # generazione del vettore latente 
#         y=self.decoder(torch.cat([z,y], dim=1))
#         return y, mu, log_sigma
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
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
    def __init__(self, architecture_yaml, info_yaml):

        self.latent_size = architecture_yaml['LATENT_SIZE']
        self.img_channels=3 
        
        self.cond_dim= self.latent_size // 4  # Dimensione della condizione
        super().__init__(architecture_yaml, info_yaml)
        self.cond_latent_proj = nn.Linear(3, self.cond_dim)  # Proiezione della condizione

    def build_encoder(self):
        return nn.Sequential(
            nn.Conv2d(self.img_channels + 3, 64, kernel_size=3, stride=2, padding=1),  # 64x32x32
            nn.ReLU(),
            ResidualBlock(64, 128, stride=2),  # 128x16x16
            ResidualBlock(128, 128, stride=1), # 128x16x16
            CBAM(128),  # Attention module
            ResidualBlock(128, 256, stride=2),  # 256x8x8
            ResidualBlock(256, 256, stride=1),  # 256x8x8
            CBAM(256),  # Attention module
            ResidualBlock(256, 512, stride=2),  # 512x4x4
            ResidualBlock(512, 512, stride=1),  # 512x4x4
            CBAM(512),  # Attention module
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 2*self.latent_size)
        )

    def build_decoder(self):
        class DecoderBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.block = nn.Sequential(
                    ResidualBlock(in_channels, in_channels),
                    ResidualBlock(in_channels, in_channels),
                    CBAM(in_channels),
                    nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            def forward(self, x):
                return self.block(x)

        model = nn.Sequential()
        size = 4
        prev = 512  # aumentato da 256 a 512

        model.append(nn.Linear(self.latent_size + self.cond_dim, prev * size * size))
        model.append(nn.ReLU())
        model.append(nn.Unflatten(1, (prev, size, size)))

        decoder_blocks = [512, 256, 128, 64, 32]
        for k in decoder_blocks:
            model.append(DecoderBlock(prev, k))
            prev = k
            size *= 2

        model.append(nn.Conv2d(prev, 3, kernel_size=3, stride=2, padding=1))
        model.append(nn.Sigmoid()) # output in [0,1]

        return model

    def forward(self, x, y):
        # skip_connections = []
        
        cond = y.unsqueeze(2).unsqueeze(3).expand(-1, -1, 64, 64)
        h=torch.cat([x,cond], dim=1)         # l'outpur dell'encoder è 2*LATENT_SIZE
        for layer in self.encoder[:]:
            h = layer(h)

        mu=h[:, :self.latent_size]    # da 0 a LATENT_SIZE corrisponde a mu
        log_sigma=h[:, self.latent_size:]     # da LATENT_SIZE fino alla fine corrisponde a log_sigma
        eps=torch.randn_like(mu)        # generazione del vettore latente secondo la normale
        z=eps*torch.exp(log_sigma)+mu       # generazione del vettore latente
        y_proj = self.cond_latent_proj(y)
        out=torch.cat([z,y_proj], dim=1)

        out = self.decoder(out)

        return out, mu, log_sigma
    
    def generate_sample(self, y_batch, save_folders):
        self.eval()
        batch_size = y_batch.shape[0]
        z = torch.randn(batch_size, *(self.latent_size,), device=self.device)
        with torch.no_grad():
            
            y_proj = self.cond_latent_proj(y_batch.to(self.device))
            image_generated = self.decoder(torch.cat([z.to(self.device),y_proj.to(self.device)], dim=1))

        image_id = 0
        last_save_folder_path = ""
        folder_to_images = {}

        for img_tensor, folder in zip(image_generated, save_folders):
            if folder != last_save_folder_path:
                last_save_folder_path = folder
                image_id = 0
            else:
                image_id += 1
            os.makedirs(folder, exist_ok=True)
            img = (img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(folder, f'sample_{image_id}.png'), img_bgr)

            # Store RGB image for plotting
            if folder not in folder_to_images:
                folder_to_images[folder] = []
            folder_to_images[folder].append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # Create figures per class
        for folder, images in folder_to_images.items():
            num_images = len(images)
            cols = min(8, num_images)
            rows = (num_images + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
            axes = np.array(axes).reshape(-1)

            for i, img in enumerate(images):
                axes[i].imshow(img)
                axes[i].axis('off')

            # Hide unused axes
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            fig_path = os.path.join(folder, 'summary_grid.png')
            plt.savefig(fig_path)
            plt.close(fig)


# ###### ResNetVAE

# class ResNetVAE(VAutoEncoder):
#     def __init__(self, architecture_yaml, info_yaml):

#         self.latent_size = architecture_yaml['LATENT_SIZE']
#         self.img_channels=3 
#         super().__init__(architecture_yaml, info_yaml)  # Proiezione della condizione
#         self.fc_mu = nn.Linear(512, self.latent_size)
#         self.fc_log_var = nn.Linear(512, self.latent_size)
#         self.fc_decoder = nn.Linear(self.latent_size + 3, 512 * 4 * 4)

#     def build_encoder(self):
#         base_model = resnet18()
#         base_model.conv1=nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         return nn.Sequential(
#             nn.Sequential(*list(base_model.children())[:-2]),  # Remove avgpool & fc
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#         )
        

#     def build_decoder(self):
#         return nn.Sequential(
#             nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8x8
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x32
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 64x64
#             nn.Sigmoid()  # output in [0,1]
#         )
        

#     def forward(self, x, y):
#         cond = y.unsqueeze(2).unsqueeze(3).expand(-1, -1, 64, 64)
#         h=torch.cat([x,cond], dim=1)       # l'outpur dell'encoder è 2*LATENT_SIZE
        
#         out = self.encoder(h)
#         mu = self.fc_mu(out)
#         log_sigma = self.fc_log_var(out)
        
#         eps=torch.randn_like(mu)        # generazione del vettore latente secondo la normale
#         z=eps*torch.exp(log_sigma)+mu       # generazione del vettore latente
        
#         z_decoder = self.fc_decoder(torch.cat([z, y], dim=1)).view(-1, 512, 4, 4)
#         out = self.decoder(z_decoder)

#         return out, mu, log_sigma

    
#     def generate_sample(self, y, save_folder, time):
#         self.eval()
#         z = torch.randn(size=(self.latent_size,))
#         with torch.no_grad():
#             image_generated = self.decoder(self.fc_decoder(torch.cat([z.unsqueeze(0).to(self.device),y.unsqueeze(0).to(self.device)], dim=1)).view(-1, 512, 4, 4))
        
#         os.makedirs(save_folder, exist_ok=True)

#         img = (image_generated.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
#         img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(os.path.join(save_folder, f'generated_{time}.png'), img_bgr)

# ##### resnet custom
# class ResNet_Mix_VAE(VAutoEncoder):
#     def __init__(self, architecture_yaml, info_yaml):

#         self.latent_size = architecture_yaml['LATENT_SIZE']
#         self.img_channels=3 
#         self.cond_dim = self.latent_size
#         super().__init__(architecture_yaml, info_yaml)  # Proiezione della condizione
#         self.fc_mu = nn.Linear(512, self.latent_size)
#         self.fc_log_var = nn.Linear(512, self.latent_size)
#         self.cond_latent = nn.Embedding(8, self.cond_dim).to(self.device)  # Embedding per le 8 classi condizionali
#         self.powers = 2 ** torch.arange(3 - 1, -1, -1).to(self.device)

#     def build_encoder(self):
#         base_model = resnet18()
#         base_model.conv1=nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         return nn.Sequential(
#             nn.Sequential(*list(base_model.children())[:-2]),  # Remove avgpool & fc
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#         )

#     def build_decoder(self):
#         class DecoderBlock(nn.Module):
#             def __init__(self, in_channels, out_channels):
#                 super().__init__()
#                 self.block = nn.Sequential(
#                     ResidualBlock(in_channels, in_channels),
#                     ResidualBlock(in_channels, in_channels),
#                     CBAM(in_channels),
#                     nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU()
#                 )
#             def forward(self, x):
#                 return self.block(x)

#         model = nn.Sequential()
#         size = 4
#         prev = 512  # aumentato da 256 a 512

#         model.append(nn.Linear(self.latent_size + self.cond_dim, prev * size * size))
#         model.append(nn.ReLU())
#         model.append(nn.Unflatten(1, (prev, size, size)))

#         decoder_blocks = [512, 256, 128, 64, 32]
#         for k in decoder_blocks:
#             model.append(DecoderBlock(prev, k))
#             prev = k
#             size *= 2

#         model.append(nn.Conv2d(prev, 3, kernel_size=3, stride=2, padding=1))
#         model.append(nn.Sigmoid())  # usa Tanh se normalizzi in [-1,1]

#         return model
        

#     def forward(self, x, y):
#         cond = y.unsqueeze(2).unsqueeze(3).expand(-1, -1, 64, 64)
#         h=torch.cat([x,cond], dim=1)       # l'outpur dell'encoder è 2*LATENT_SIZE
        
#         out = self.encoder(h)
#         mu = self.fc_mu(out)
#         log_sigma = self.fc_log_var(out)
        
#         eps=torch.randn_like(mu)        # generazione del vettore latente secondo la normale
#         z=eps*torch.exp(log_sigma)+mu       # generazione del vettore latente
        
#         integers = (y * self.powers).sum(dim=1).long()
#         y_decoder = self.cond_latent(integers)
#         # breakpoint()
#         z_decoder = torch.cat([z, y_decoder], dim=1)
#         out = self.decoder(z_decoder)

#         return out, mu, log_sigma
    
#     def generate_sample(self, y, save_folder, time):
#         self.eval()
#         z = torch.randn(size=(self.latent_size,)).unsqueeze(0).to(self.device)
#         y = y.unsqueeze(0).to(self.device)
        
#         integers = (y * self.powers).sum(dim=1).long()
#         y_decoder = self.cond_latent(integers)
#         with torch.no_grad():
#             image_generated = self.decoder(torch.cat([z,y_decoder], dim=1))
        
#         os.makedirs(save_folder, exist_ok=True)

#         img = (image_generated.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
#         img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(os.path.join(save_folder, f'generated_{time}.png'), img_bgr)
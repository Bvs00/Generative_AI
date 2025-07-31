import torch
import torch.nn as nn
import math
import os
import yaml
import cv2
from cbam import CBAM
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils_blocks import *
import matplotlib.pyplot as plt
import numpy as np

class UNetBlock(nn.Module):
      def __init__(self, size, time_encoding_size, outer_features, inner_features, cond_features, inner_block=None):
          super().__init__()
          self.time_encoding_size = time_encoding_size
          self.size = size
          self.outer_features = outer_features
          self.inner_features = inner_features
          self.cond_features = cond_features
          self.encoder = self.build_encoder(outer_features+cond_features, inner_features)
          self.decoder = self.build_decoder(inner_features+cond_features+self.time_encoding_size, outer_features)
          self.combiner = self.build_combiner(2*outer_features, outer_features)
          self.inner = inner_block

      def forward(self, x, time_encodings, cond):
          x0=x
          cc=cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, self.size, self.size).to(device=x.device)
          x=torch.cat((x,cc), dim=1)
          y=self.encoder(x)
          if self.inner:
              y=self.inner(y, time_encodings, cond)
          half_size=self.size//2
          cc=cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, half_size, half_size).to(device=x.device)
          tt=time_encodings.view(-1, self.time_encoding_size, 1, 1).expand(-1, -1, half_size, half_size).to(device=x.device)
          y1=torch.cat((y,cc,tt), dim=1)
          x1=self.decoder(y1)
          x2=torch.cat((x1,x0), dim=1)
          return self.combiner(x2)

      def build_combiner(self, from_features, to_features):
          return nn.Conv2d(from_features, to_features, 1)

      def build_encoder(self, from_features, to_features):
          model=nn.Sequential(
                  nn.Conv2d(from_features, from_features, 5, padding='same', bias=False),
                  nn.BatchNorm2d(from_features),
                  nn.ReLU(),
                  CBAM(from_features),  # Attention module
                  nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
                  nn.BatchNorm2d(from_features),
                  nn.ReLU(),
                  CBAM(from_features),  # Attention module
                  nn.Conv2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(to_features),
                  nn.ReLU()
          )
          return model

      def build_decoder(self, from_features, to_features):
          model=nn.Sequential(
                  nn.Conv2d(from_features, from_features, 5, padding='same', bias=False),
                  nn.BatchNorm2d(from_features),
                  nn.ReLU(),
                  CBAM(from_features),  # Attention module  
                  nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
                  nn.BatchNorm2d(from_features),
                  nn.ReLU(),
                  CBAM(from_features),  # Attention module                
                  nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(to_features),
                  nn.ReLU()
          )
          return model

# class UNetBlock_film(nn.Module):
#     def __init__(self, size, time_encoding_size, outer_features, inner_features, cond_features, inner_block=None):
#         super().__init__()
#         self.time_encoding_size = time_encoding_size
#         self.size = size
#         self.outer_features = outer_features
#         self.inner_features = inner_features
#         self.cond_features = cond_features
#         self.encoder = self.build_encoder(outer_features, inner_features)
#         self.decoder = self.build_decoder(inner_features+self.time_encoding_size, outer_features)
#         self.combiner = self.build_combiner(2*outer_features, outer_features)
#         self.inner = inner_block

#     def forward(self, x, time_encodings, cond):
#         x0=x
#         y=self.encoder(x, cond)
#         if self.inner:
#             y=self.inner(y, time_encodings, cond)
#         half_size=self.size//2
#         tt=time_encodings.view(-1, self.time_encoding_size, 1, 1).expand(-1, -1, half_size, half_size).to(device=x.device)
#         y1=torch.cat((y,tt), dim=1)
#         x1=self.decoder(y1, cond)
#         x2=torch.cat((x1,x0), dim=1)
#         return self.combiner(x2)

#     def build_combiner(self, from_features, to_features):
#         return nn.Conv2d(from_features, to_features, 1)

#     def build_encoder(self, from_features, to_features):
#         return EncoderWithFilm(from_features=from_features, to_features=to_features, cond_features=self.cond_features)

#     def build_decoder(self, from_features, to_features):
#         return DecoderWithFiLM(from_features=from_features, to_features=to_features, cond_features=self.cond_features)


class Network(nn.Module):
    def __init__(self, architecture_yaml, train_yaml):
        super().__init__()

        # Load YAML files
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture_yaml=architecture_yaml
        self.train_yaml=train_yaml

        # Extract parameters from the YAML files
        # trianing parameters
        self.num_epochs = self.train_yaml["TRAINING"]["NUM_EPOCHS"]
        self.L = self.train_yaml["TRAINING"]["L"]
        self.cond_shape = self.train_yaml["TRAINING"]["COND_SHAPE"]
        self.scheduleType = self.train_yaml["TRAINING"].get("SCHEDULE_TYPE", None)
        # architecture parameters
        self.time_encodig_size = self.architecture_yaml["TIME_ENCODING_SIZE"]
        self.time_encoding=TimeEncoding(self.L, self.time_encodig_size)
        self.feat_list = self.architecture_yaml["FEAT_LIST"]
        self.unet_block_class_name = self.architecture_yaml["UNET_BLOCK"]

        # Initialize UNet block class from the class name
        self.unet_block_class = globals().get(self.unet_block_class_name, None)
        if self.unet_block_class is None:
            print(f"[WARNING] UNET_BLOCK class '{class_name}' not found. Using default class '{UNetBlock.__name__}' instead.")
            self.unet_block_class = UNetBlock

        # Initialize the network components
        self.pre=nn.Sequential(
            nn.Conv2d(3, self.feat_list[0], 3, padding='same'),
            nn.ReLU())
        self.unet = self.build_unet(64, self.feat_list)
        self.post=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.feat_list[0], 3, 3, padding='same'))
        
        # Set utility parameters
        self.image_dimensions=(1,1,1)

        # one-hot encoding for the conditioning and powers for the integer representation
        self.cond_one_hot=torch.eye(self.cond_shape[0], device=device)
        self.powers = 2 ** torch.arange(3 - 1, -1, -1).to(self.device)
        print("Network initialized correctly.")

    # Set training parameters after the network is initialized
    def set_trining_parameters(self):
        self.noise_schedule=NoiseSchedule(self.L)
        self.loss_function=nn.MSELoss()
        lr = self.train_yaml["TRAINING"]["LR"]
        self.optimizer=torch.optim.AdamW(self.parameters(), lr=lr)
        if self.scheduleType != None:
            if self.scheduleType == "StepLR":
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
            elif self.scheduleType == "CyclicLR":
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr*0.01, max_lr=lr)
            print(f"Using schedule type: {self.scheduleType}")

    
    def forward(self, x, t, cond):
        # get the time encoding for the current step
        enc=self.time_encoding[t]
        # forward pass through the network
        x=self.pre(x)
        y=self.unet(x, enc, cond)
        y=self.post(y)
        return y

    def build_unet(self, size, feat_list):
        if len(feat_list)>2:
            inner_block=self.build_unet(size//2, feat_list[1:])
        else:
            inner_block=None
        return self.unet_block_class(size, self.time_encodig_size, feat_list[0], feat_list[1], self.cond_shape[0], inner_block)

    def training_epoch(self, checkpoint_path, dataloader):
        self.train()
        list_loss = []
        start_epoch=0
        print("START TRAINING")
        existing_checkpoint = None
        checkpoint_filename, _ = os.path.splitext(os.path.basename(checkpoint_path))
        checkpoint_folder = os.path.dirname(checkpoint_path)
        existing_checkpoint = find_matching_checkpoint(checkpoint_folder, checkpoint_filename)

        if existing_checkpoint:
            checkpoint = torch.load(existing_checkpoint, map_location=device)

            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduleType is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                self.scheduler = None
            start_epoch = checkpoint['epoch']
            list_loss = checkpoint['list_loss']
            print(f"Checkpoint path: {existing_checkpoint}")
            print(f"Riprendo l'addestramento da epoca {start_epoch+1}...")
        
        for epoch in range(start_epoch, start_epoch + self.num_epochs, 1):
            loss = self.train_one_epoch(dataloader)
            print(f'Epoch: {epoch+1}\n\tLoss: {loss}')
            list_loss.append(loss)
        
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'list_loss': list_loss,
            }
            if self.scheduleType is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()


            # Split the base path into directory, filename, and extension
            base_dir = os.path.dirname(checkpoint_path)
            base_name = os.path.basename(checkpoint_path)
            name, ext = os.path.splitext(base_name)
            
            os.makedirs(base_dir, exist_ok=True)

            # Build epoch-specific checkpoint filenames
            current_checkpoint = os.path.join(base_dir, f"{name}_epoch_{epoch + 1}{ext}")
            previous_checkpoint = os.path.join(base_dir, f"{name}_epoch_{epoch}{ext}")

            # Remove previous checkpoint if it exists
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)

            torch.save(checkpoint, current_checkpoint)
        print("FINISH TRANING")


    def train_one_epoch(self, dataloader):
        self.train()
        average_loss=0.0
        for x, y in tqdm(dataloader, dynamic_ncols=True):
            x=x.to(device=device)
            n=x.shape[0] # Minibatch size
            inty = (y.to(device) * self.powers).sum(dim=1).long()
            cond=self.cond_one_hot[inty]

            # Remove the conditioning information with probability P=0.2
            P=0.2
            u=torch.rand((n,))
            cond[u<P,:]=0.0

            # Generate the random step indices (one for each sample in the minibatch)
            t=torch.randint(0, self.L, (n,), device=device)

            # Generate the random noise
            eps=torch.randn_like(x)

            # Compute latent image
            sqrt_alpha=self.noise_schedule.sqrt_alpha[t].view(-1, * self.image_dimensions)
            sqrt_1_alpha=self.noise_schedule.sqrt_1_alpha[t].view(-1, * self.image_dimensions)

            # diffusion kernel
            zt=sqrt_alpha*x + sqrt_1_alpha*eps

            # Compute the output of the network (estimate of eps)
            g=self(zt,t,cond)
            # Compute the loss
            loss=self.loss_function(g, eps)

            # set the gradients to zero, backpropagate and update the parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduleType is not None and self.scheduleType == "CyclicLR": # in case of cycllic decay, step the scheduler after each batch
                self.scheduler.step()
            average_loss=0.9*average_loss+0.1*loss.cpu().item()

        if self.scheduleType is not None and self.scheduleType == "StepLR": # in case of step decay, step the scheduler after each epoch
            self.scheduler.step()
        return average_loss

    def generate_sample(self, y_batch, save_folders, lam=1.5):
        self.eval()
        with torch.no_grad():
            batch_size = y_batch.shape[0]

            z = torch.randn(batch_size, *(3,64,64), device=device)

            # Compute condition indices and one-hot encodings
            inty = (y_batch.to(device) * self.powers).sum(dim=1).long() # shape: (B,)
            cond=self.cond_one_hot[inty]
            cond0=torch.zeros_like(cond)

            for kt in reversed(range(self.L)):
                t=torch.tensor(kt).view(1).expand(batch_size)

                # Prepare the coefficients to the right shape
                beta=self.noise_schedule.beta[t].view(-1, *self.image_dimensions)
                sqrt_1_alpha=self.noise_schedule.sqrt_1_alpha[t].view(-1, *self.image_dimensions)
                sqrt_1_beta=self.noise_schedule.sqrt_1_beta[t].view(-1, *self.image_dimensions)
                sqrt_beta=self.noise_schedule.sqrt_beta[t].view(-1, *self.image_dimensions)

                # Estimate the error
                g1 = self(z, t, cond)
                g0 = self(z, t, cond0)
                g = lam*g1 + (1-lam)*g0

                # Compute mu
                mu=(z-beta/sqrt_1_alpha*g)/sqrt_1_beta

                # Generate and add the error
                if kt>0:
                    eps=torch.randn_like(z)
                    z=mu+sqrt_beta*eps
                else:
                    z=mu

        image_id = 0
        last_save_folder_path = ""
        folder_to_images = {}
        for img_tensor, folder in zip(z, save_folders):
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

        return z





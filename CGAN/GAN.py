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
from torchvision.models import resnet18, ResNet18_Weights
import yaml
from torch.nn.functional import binary_cross_entropy
import sys
import numpy as np

class Generator(nn.Module):
    "Assumes that the input images are 64x64"
    def __init__(self, architecture_yaml, train_yaml):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.latent_size = architecture_yaml["GENERATOR"]["LATENT_SIZE"]
        self.lr_gen = train_yaml["TRAINING"]["GENERATOR"]["LR"]

        self.build_generator()
    
    def gen_loss_function(self, d_synth):
        t_synth=torch.ones_like(d_synth)
        return binary_cross_entropy(d_synth,t_synth)
    
    @abstractmethod
    def forward(self, x, y):
        raise NotImplementedError("Must be implemented in subclass")
 
    @abstractmethod
    def build_generator(self):
        raise NotImplementedError("Must be implemented in subclass")
    
    def generate_sample(self, y_batch, save_folders):
        self.eval()
        batch_size = y_batch.shape[0]
        z = torch.randn(batch_size, *(self.latent_size,), device=self.device)
        with torch.no_grad():
            
            image_generated = self(z.to(self.device),y_batch.to(self.device))

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

class Discriminator(nn.Module):
    "Assumes that the input images are 64x64"
    def __init__(self, architecture_yaml, train_yaml):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_smoothing = train_yaml["TRAINING"]["LABEL_SMOOTHING"]
        self.lr_disc = train_yaml["TRAINING"]["DISCRIMINATOR"]["LR"]

        self.build_discriminator()
    
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

        # Initialize optimizers
        self.gen_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=self.generator.lr_gen)
        self.disc_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=self.discriminator.lr_disc)
        
        # Initialize learning rate schedulers for discriminator
        #self.disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.disc_optimizer, T_max=self.num_epochs, eta_min=self.discriminator.lr_disc/100)
    
    def training_epoch(self, checkpoint_path, dataloader):
        self.generator.train()
        self.discriminator.train()
        list_loss_generator = []
        list_loss_discriminator = []
        list_sum_dtrue = []
        list_sum_dsynth = []
        start_epoch=0
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

            self.generator.load_state_dict(checkpoint['model_state_dict_generator'])
            self.discriminator.load_state_dict(checkpoint['model_state_dict_discriminator'])
            self.gen_optimizer.load_state_dict(checkpoint['optimizer_state_dict_generator'])
            self.disc_optimizer.load_state_dict(checkpoint['optimizer_state_dict_discriminator'])
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
            
            # update the learning rate scheduler for discriminator
            #self.disc_scheduler.step()
        
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict_generator': self.generator.state_dict(),
                'model_state_dict_discriminator': self.discriminator.state_dict(),
                'optimizer_state_dict_generator': self.gen_optimizer.state_dict(),
                'optimizer_state_dict_discriminator': self.disc_optimizer.state_dict(),
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
        for x_true, cls in tqdm(dataloader):
        # for x_true, cls in dataloader:
            x_true=x_true.to(self.device)
            cls = cls.to(self.device)
            d_true=self.discriminator(x_true, cls)
            z=torch.randn(x_true.shape[0], self.generator.latent_size, device=self.device)
            x_synth=self.generator(z, cls)
            d_synth = self.discriminator(x_synth, cls)

            # update the discriminator
            self.disc_optimizer.zero_grad()
            dloss=self.discriminator.disc_loss_function(d_true,d_synth)
            dloss.backward(retain_graph=True)
            self.disc_optimizer.step()

            #update the generator
            d_synth=self.discriminator(x_synth, cls)
            self.gen_optimizer.zero_grad()
            gloss=self.generator.gen_loss_function(d_synth)
            gloss.backward()
            self.gen_optimizer.step()

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
        self.model=nn.Sequential()
        for k in self.discriminator_channel_progression:
            self.model.append(nn.Conv2d(self.initial_channel_dim, k, 3, padding='same'))
            nn.BatchNorm2d(k),
            self.model.append(nn.LeakyReLU())
            self.model.append(nn.Conv2d(k, k, 3, stride=2, padding=1))
            nn.BatchNorm2d(k),
            self.model.append(nn.LeakyReLU())
            self.initial_channel_dim=k
            self.wh_size=self.wh_size//2    # al secondo Conv2D viene dimezzata la dimensione di self.wh_size
        self.model.append(nn.Flatten())
        features=k*2*self.wh_size    # numero di features a valle del flatten
        self.model.append(nn.Linear(features, k))
        self.model.append(nn.LeakyReLU())
        self.model.append(nn.Dropout(p=0.3))
        self.model.append(nn.Linear(k, 1))
        self.model.append(nn.Sigmoid())
        
        # assert size==8
        # assert k==128

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
        self.model=nn.Sequential()
        size=4
        prev=self.generator_channel_progression[0]
        self.model.append(nn.Linear(self.latent_size + self.latent_size//8, (prev//2)*size*size))
        self.model.append(nn.GELU())
        self.model.append(nn.Linear((prev//2)*size*size, prev*size*size))
        self.model.append(nn.GELU())
        self.model.append(nn.Unflatten(1, (prev, size, size)))
        for k in self.generator_channel_progression:
            self.model.append(nn.Conv2d(prev, k, 3, padding='same'))
            self.model.append(nn.BatchNorm2d(k))
            self.model.append(nn.GELU())
            self.model.append(nn.Conv2d(k, k, 3, padding='same'))
            self.model.append(nn.BatchNorm2d(k))
            self.model.append(nn.GELU())
            self.model.append(nn.ConvTranspose2d(k, k, 3, stride=2, padding=1,
                                            output_padding=1))
            self.model.append(nn.GELU())
            prev=k
            size=size*2
        # assert size==128
        # assert k==32
        self.model.append(nn.Conv2d(k, k//2, 3, padding='same'))
        self.model.append(nn.GELU())
        self.model.append(nn.Conv2d(k//2, 3, 3, padding='same'))
        self.model.append(nn.Sigmoid())

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
        self.model = nn.Sequential(
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

class ResNetVDiscriminator_2p(Discriminator):
    def __init__(self, label_smoothing, lr_disc):
        # check if self.initial_channel_dim is already defined (in an inherited class)
        if 'initial_channel_dim' not in self.__dict__:
            print("initial_channel_dim not defined, setting to default value of 6")
            self.initial_channel_dim = 3 + 3  # prev è il numero di feature maps iniziale (nel caso di condizionale ad 8 classi dobbiamo avere 3-RGB + 3-Classi = 6)
        
        super().__init__(label_smoothing, lr_disc)
        self.cond_latent = nn.Embedding(8, 64).to(self.device)  # Embedding per le 8 classi condizionali
        self.powers = 2 ** torch.arange(3 - 1, -1, -1).to(self.device)

    def build_discriminator(self):
        base_model = resnet18()
        base_model.conv1=nn.Conv2d(self.initial_channel_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net1 = nn.Sequential(
            nn.Sequential(*list(base_model.children())[:-2]),  # Remove avgpool & fc
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.net2 = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        BATCH_SIZE = x.shape[0]
        # compute the variance of x pixel by pixel among the batch dimension and the channel dimension
        # x.shape = (BATCH_SIZE, C, H, W)
        x_var = torch.var(x, dim=0, keepdim=True).expand(BATCH_SIZE, -1, -1, -1)  # expand to match the input shape
        # concatenate the variance channel to the input
        x = torch.cat([x, x_var], dim=1)


        res_out = self.net1(x)

        integers = (c * self.powers).sum(dim=1).long()
        c_token = self.cond_latent(integers)
        out = self.net2(torch.cat([res_out, c_token], dim=1))
        return out

class ResNetVDiscriminator(ResNetDiscriminator):
    def __init__(self, architecture_yaml, train_yaml):
        self.initial_channel_dim = 3 + 3 + 3     # prev è il numero di feature maps iniziale (nel caso di condizionale ad 8 classi dobbiamo avere 3-RGB + 3-Classi = 6)
        super().__init__(architecture_yaml, train_yaml)
    
    def forward(self, x, c):
        BATCH_SIZE = x.shape[0]
        # compute the variance of x pixel by pixel among the batch dimension and the channel dimension
        # x.shape = (BATCH_SIZE, C, H, W)
        x_var = torch.var(x, dim=0, keepdim=True).expand(BATCH_SIZE, -1, -1, -1)  # expand to match the input shape
        # concatenate the variance channel to the input
        x = torch.cat([x, x_var], dim=1)

        out = super().forward(x, c)
        return out

class ResNetVDiscriminator_up(Discriminator):
    def __init__(self, architecture_yaml, train_yaml, cond_dim=3):
        self.cond_dim = cond_dim
        self.image_feature_dim = 512
        self.variance_feature_dim = 64
        self.condition_feature_dim = 128 
        super().__init__(architecture_yaml, train_yaml)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(self.device)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(self.device)

        # Initialize variance tensor
        self.v_powers = 2 ** torch.arange(3, device=self.device, dtype=torch.int8)

    def _normalize_for_resnet(self, x):
        return (x - self.imagenet_mean) / self.imagenet_std

    def build_discriminator(self):
        # Load pretrained resnet
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_backbone = nn.Sequential(*list(base_model.children())[:-2])

        # Conditioning vector branch (MLP)
        self.condition_branch = nn.Sequential(
            nn.Linear(self.cond_dim, self.condition_feature_dim//2),
            nn.ReLU(),
            nn.Linear(self.condition_feature_dim//2, self.condition_feature_dim),
            nn.ReLU()
        )
        
        # Variance branch (CNN)
        self.variance_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (B, 1, 64, 64) → (B, 16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # → (B, 32, 16, 16)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # → (B, 32)
            nn.Linear(32, self.variance_feature_dim),
            nn.ReLU()
        )

        # Final head
        total_feat_dim = self.image_feature_dim + self.condition_feature_dim + self.variance_feature_dim
        self.head = nn.Sequential(
            nn.Linear(total_feat_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _compute_intra_class_variance(self, x, c):
        # x: (B, C, H, W)
        # c: (B, D), binary (int or float), on the same device as x
        B, C, H, W = x.shape
        c_flat = c.to(torch.int8)


        # Binary hash (base-2)
        condition_ids = (c_flat * self.v_powers).sum(dim=1)

        # Unique group IDs and indices
        unique_ids, inverse_indices = torch.unique(condition_ids, return_inverse=True)
        num_classes = unique_ids.shape[0]

        x_var = torch.zeros((B, 1, H, W), device=self.device)

        for class_idx in range(num_classes):
            mask = (inverse_indices == class_idx)
            selected = mask.nonzero(as_tuple=True)[0]
            if selected.numel() > 1:
                x_group = x[selected]
                var = torch.var(x_group, dim=0, correction=0)  # Not using Bessel's correction (1)
            else:
                var = torch.zeros((C, H, W), device=x.device)
            var_mean = var.mean(dim=0, keepdim=True)
            x_var[mask] = var_mean
        return x_var

    def forward(self, x, c):
        B, _, H, W = x.shape
        device = x.device

        # --- Compute intra-class variance ---
        x_var = self._compute_intra_class_variance(x, c)  # (B, 1, H, W) update x_var

        # --- ResNet (image branch) ---
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x_resized = self._normalize_for_resnet(x_resized)  # Normalize for ResNet
        image_features = self.resnet_backbone(x_resized)
        image_features = F.adaptive_avg_pool2d(image_features, (1, 1)).flatten(1)  # (B, 512)

        # --- Condition branch ---
        cond_features = self.condition_branch(c)  # (B, condition_feature_dim)

        # --- Variance branch ---
        variance_features = self.variance_branch(x_var)  # (B, variance_feature_dim)
        
        # --- Combine all ---
        combined = torch.cat([image_features, cond_features, variance_features], dim=1)
        out = self.head(combined)
        return out

class ResNetVDiscriminator_up_opt(Discriminator):
    def __init__(self, architecture_yaml, train_yaml, cond_dim=3):
        self.cond_dim = cond_dim
        self.image_feature_dim = 512
        self.variance_feature_dim = 64 # self.image_feature_dim//8 
        self.condition_feature_dim = 11 # near the image_feature_dim//16 (11*3=33, 33 is close to 32)
        super().__init__(architecture_yaml, train_yaml)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(self.device)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(self.device)

        # Initialize variance tensor
        self.v_powers = 2 ** torch.arange(3, device=self.device, dtype=torch.int8)

    def _normalize_for_resnet(self, x):
        return (x - self.imagenet_mean) / self.imagenet_std

    def build_discriminator(self):
        # Load pretrained resnet
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_backbone = nn.Sequential(*list(base_model.children())[:-2])

        self.variance_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # (B, 1, H, W) → (B, 1, 8, 8)
            nn.Flatten(),  # → (B, 1*8*8)
        )

        # Final head
        total_feat_dim = self.image_feature_dim + 3*self.condition_feature_dim + self.variance_feature_dim
        self.head = nn.Sequential(
            nn.Linear(total_feat_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _compute_intra_class_variance(self, x, c):
        # x: (B, C, H, W)
        # c: (B, D), binary (int or float), on the same device as x
        B, C, H, W = x.shape
        c_flat = c.to(torch.int8)


        # Binary hash (base-2)
        condition_ids = (c_flat * self.v_powers).sum(dim=1)

        # Unique group IDs and indices
        unique_ids, inverse_indices = torch.unique(condition_ids, return_inverse=True)
        num_classes = unique_ids.shape[0]

        x_var = torch.zeros((B, 1, H, W), device=self.device)

        for class_idx in range(num_classes):
            mask = (inverse_indices == class_idx)
            selected = mask.nonzero(as_tuple=True)[0]
            if selected.numel() > 1:
                x_group = x[selected]
                var = torch.var(x_group, dim=0, correction=0)  # Not using Bessel's correction (1)
            else:
                var = torch.zeros((C, H, W), device=x.device)
            var_mean = var.mean(dim=0, keepdim=True)
            x_var[mask] = var_mean
        return x_var

    def forward(self, x, c):
        B, _, H, W = x.shape
        device = x.device

        # --- Compute intra-class variance ---
        x_var = self._compute_intra_class_variance(x, c)  # (B, 1, H, W) update x_var

        # --- ResNet (image branch) ---
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x_resized = self._normalize_for_resnet(x_resized)  # Normalize for ResNet
        image_features = self.resnet_backbone(x_resized)
        image_features = F.adaptive_avg_pool2d(image_features, (1, 1)).flatten(1)  # (B, 512)

        # --- Condition branch ---
        # cond_features = self.condition_branch(c)  # (B, condition_feature_dim)
        cond_features = c.repeat_interleave(self.condition_feature_dim, dim=1)  # (B, image_feature_dim)
        # --- Variance branch ---
        variance_features = self.variance_branch(x_var)  # (B, variance_feature_dim)
        # --- Combine all ---
        combined = torch.cat([image_features, cond_features, variance_features], dim=1)
        out = self.head(combined)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class ImprovedGenerator(Generator):
    def __init__(self, architecture_yaml, train_yaml):
        self.generator_channel_progression = architecture_yaml["GENERATOR"]['CHANNEL_PROGRESSION']
        super().__init__(architecture_yaml, train_yaml)
        self.cond_latent = nn.Embedding(8, self.latent_size // 8).to(self.device)
        self.powers = 2 ** torch.arange(3 - 1, -1, -1).to(self.device)

        self.model = self.build_generator()

    def build_generator(self):
        layers = nn.ModuleList()
        size = 4
        prev = self.generator_channel_progression[0]

        # Dense + reshape
        self.fc = nn.Sequential(
            nn.Linear(self.latent_size + self.latent_size // 8, (prev // 2) * size * size),
            nn.GELU(),
            nn.Linear((prev // 2) * size * size, prev * size * size),
            nn.GELU()
        )

        self.unflatten = nn.Unflatten(1, (prev, size, size))

        self.conv_blocks = nn.ModuleList()
        for k in self.generator_channel_progression:
            block = nn.Sequential(
                nn.Conv2d(prev, k, 3, padding=1),
                nn.BatchNorm2d(k),
                nn.GELU(),
                ResidualBlock(k),
                nn.ConvTranspose2d(k, k, 3, stride=2, padding=1, output_padding=1),
                nn.GELU()
            )
            self.conv_blocks.append(block)
            prev = k
            size *= 2

        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(prev, prev // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(prev // 2, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, c):
        integers = (c * self.powers).sum(dim=1).long()
        c_token = self.cond_latent(integers)
        zc = torch.cat([z, c_token], dim=1)

        x = self.fc(zc)
        x = self.unflatten(x)

        for block in self.conv_blocks:
            x = block(x)

        return self.final(x)
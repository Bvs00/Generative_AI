import os
from tqdm import tqdm
os.environ["MPLCONFIGDIR"] = os.path.expanduser("/tmp/matplotlib")
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CelebA
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
import argparse
from CelebA import CelebADataset
import random
import numpy as np

LATENT_SIZE = 32
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class AutoEncoder(nn.Module):
    "Assumes that the input images are 64x64"
    def __init__(self, args):
        super().__init__()
        self.latent_size=args.latent_size
        self.encoder=self.build_encoder(self.latent_size)
        self.decoder=self.build_decoder(self.latent_size)
 
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
 
    def build_encoder(self, latent_size):
        model=nn.Sequential()
        prev=3 + 3      # prev è il numero di feature maps iniziale (nel caso di condizionale ad 8 classi dobbiamo avere 3-RGB + 3-Classi = 6)
        size=64    # size identifica la dimensione dell'immagine di input (celebA è 64)
        for k in [32, 64, 128, 128]:
            model.append(nn.Conv2d(prev, k, 3, padding='same'))
            model.append(nn.ReLU())
            model.append(nn.Conv2d(k, k, 3, stride=2, padding=1))
            model.append(nn.ReLU())
            prev=k
            size=size//2    # al secondo Conv2D viene dimezzata la dimensione di size
        model.append(nn.Flatten())
        features=k*size*size    # numero di features a valle del flatten
        model.append(nn.Linear(features, 2*latent_size))
        # assert size==8
        # assert k==128
        return model
 
    def build_decoder(self, latent_size):
        model=nn.Sequential()
        size=4
        prev=128
        model.append(nn.Linear(latent_size + 3, prev*size*size))
        model.append(nn.ReLU())
        model.append(nn.Unflatten(1, (prev, size, size)))
        for k in [128, 128, 64, 32]:
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
    
    def generate_sample(self, y):
        self.eval()
        z = torch.randn(size=(args.latent_size,))
        with torch.no_grad():
            image_generated = self.decoder(torch.cat([z.unsqueeze(0).to(device),y.unsqueeze(0).to(device)], dim=1))
        
        plt.figure()
        plt.imshow(image_generated[0][0].cpu(), cmap='gray')
        plt.savefig(f"Image_Generated/CVAE_{y}")


######### LOSS ############
reconstruction_loss_function = nn.BCELoss(reduction='sum')


def kl_loss_function(mu, log_sigma):
    log_sigma_2 = 2*log_sigma
    kl = 0.5*(mu**2+torch.exp(log_sigma_2)-1-log_sigma_2)
    return torch.sum(kl)

def loss_function(reconstructed, original, mu, log_sigma):
    return reconstruction_loss_function(reconstructed, original) + \
        args.beta*kl_loss_function(mu, log_sigma)


##############  TRAIN     ###############

def training_epoch(model, criterion, optimizer, dataloader):
    for epoch in range(args.num_epochs):
        model.train()
        average_loss = 0.0
        list_loss_train = []
        
        for x_batch, label_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", dynamic_ncols=True):
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)
            
            output, mu, log_sigma = model(x_batch, label_batch)
            
            loss = criterion(output, x_batch, mu, log_sigma)
            
            loss.backward()
            optimizer.step()
            
            average_loss += loss
        print(f"Epoch {epoch+1} completed. Average loss = {average_loss/len(dataloader)}")
        list_loss_train.append(average_loss/len(dataloader))
        
        checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'list_loss_train': list_loss_train,
        }
        torch.save(checkpoint, f"{args.name_model}_checkpoint.pth")
    
    # plot loss train
    torch.save(model.state_dict(),f"{args.name_model}_best_model.pth")

# 15 eyglasses
# 16 Goatee
# 20  Male
# 22  Mustache
# 24  No_Beard
def plot_img(img):
    plt.imshow(img)
    plt.savefig('prova')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--name_model', type=str, default="VAE")
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()
    
    args.name_model = f"{args.name_model}_{args.latent_size}"
    print(device)
    random.seed(1)  # Set seed for Python's random
    np.random.seed(1)  # Set seed for NumPy
    torch.manual_seed(1)  # Set seed for PyTorch
    
    transform=transforms.Compose([
        transforms.CenterCrop((160,160)),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])
    
    # training_set = CelebA(root='./celeba', transform=transform, download=False, split='train')
    dataset = CelebA(root="/mnt/datasets/eeg/celeba", split='all', transform=transform, target_type="attr", download=False)
    
    training_loader = DataLoader(CelebADataset(dataset), batch_size=128, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True)
    
    model = AutoEncoder()
    model=model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    training_epoch(model, loss_function, optimizer, training_loader)
        
    breakpoint()
    
    
    
    

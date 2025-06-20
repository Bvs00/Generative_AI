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
from VAE import AutoEncoder


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
    start_epoch = 0
    list_loss_train = []
    
    if args.checkpoint and os.path.exists(f"{args.name_model}_checkpoint.pth"):
        checkpoint = torch.load(f"{args.name_model}_checkpoint.pth")

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        list_loss_train = checkpoint['list_loss_train']
        print(f"Riprendo l'addestramento da epoca {start_epoch+1}...")
    
    for epoch in range(start_epoch, args.num_epochs, 1):
        model.train()
        average_loss = 0.0

        # for x_batch, label_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", dynamic_ncols=True):
        for x_batch, label_batch in dataloader:
            optimizer.zero_grad()
            x_batch = x_batch.to(args.device)
            label_batch = label_batch.to(args.device)
            
            output, mu, log_sigma = model(x_batch, label_batch)
            
            loss = criterion(output, x_batch, mu, log_sigma)
            
            loss.backward()
            optimizer.step()
            
            average_loss += loss
        print(f"Epoch {epoch+1} completed. Average loss = {average_loss/(len(dataloader)*1000)}")
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
    print("FINISH TRANING")


def plot_img(img):
    plt.imshow(img)
    plt.savefig('prova')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--name_model', type=str, default="Results_VAE_CelebA/VAE")
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-checkpoint', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    saved_path = args.name_model.split('/')[:-1]
    saved_path = '/'.join(saved_path)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    
    args.name_model = f"{args.name_model}_{args.latent_size}"
    print(args.device)
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
    
    model = AutoEncoder(args=args)
    model=model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    training_epoch(model, loss_function, optimizer, training_loader)
    
    

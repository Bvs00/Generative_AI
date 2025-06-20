import os
os.environ["MPLCONFIGDIR"] = os.path.expanduser("/tmp/matplotlib")
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from matplotlib import pyplot as plt
import argparse

NUM_EPOCHS = 50
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class AutoEncoder(nn.Module):
    def __init__(self, dims=[128, 64, 2], num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear((28*28) + num_classes, dims[0]), # input 
            nn.ReLU(),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU()
        )
        self.linear_mu=nn.Linear(dims[1], dims[2])    # 2 per visualizzazione su un piano bidimensionale
        self.linear_log_sigma=nn.Linear(dims[1], dims[2])     # 2 per visualizzazione su un piano bidimensionale

        self.decoder = nn.Sequential(
            nn.Linear(dims[2] + num_classes, dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[0]),
            nn.ReLU(),
            nn.Linear(dims[0], 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1,28,28))
        )
    
    def forward(self, x, y):
        x = torch.reshape(x, (x.shape[0], -1))
        out=self.encoder(torch.cat([x, y], dim=1))     # generazione del vettore latente
        mu=self.linear_mu(out)      # generazione del vettore mu
        log_sigma=self.linear_log_sigma(out)    #generazione del vettore log_sigma

        eps=torch.randn_like(log_sigma)         # generazione di un vettore secondo la normale (0,1)
        latent_vector=mu+eps*torch.exp(log_sigma)       # reparametrization trick

        reconstruction_vector = self.decoder(torch.cat([latent_vector, y], dim=1))                 
        return reconstruction_vector, mu, log_sigma

    def compute_latent_vectors(self, x, y):
        one_hot_matrix = torch.eye(10)
        out=self.encoder(x, one_hot_matrix[y])
        mu=self.linear_mu(out)
        log_sigma=self.linear_log_sigma(out)
        eps=torch.randn_like(log_sigma)
        latent_vector=mu+eps*torch.exp(log_sigma)
        return latent_vector
    
    def generate_sample(self, y):
        self.eval()
        one_hot_matrix = torch.eye(10)
        z = torch.randn(size=(2,))
        with torch.no_grad():
            image_generated = self.decoder(torch.cat([z.unsqueeze(0).to(device), one_hot_matrix[y].unsqueeze(0).to(device)], dim=1))
        
        plt.figure()
        plt.imshow(image_generated[0][0].cpu(), cmap='gray')
        plt.savefig(f"Image_Generated/CVAE_{y+1}")


######### LOSS ############
reconstruction_loss_function = nn.BCELoss(reduction='sum')
beta=0.7

def kl_loss_function(mu, log_sigma):
    log_sigma_2 = 2*log_sigma
    kl = 0.5*(mu**2+torch.exp(log_sigma_2)-1-log_sigma_2)
    return torch.sum(kl)

def loss_function(reconstructed, original, mu, log_sigma):
    return reconstruction_loss_function(reconstructed, original) + \
        beta*kl_loss_function(mu, log_sigma)


##############  TRAIN     ###############

def training_epoch(model, criterion, optimizer, dataloader):
    one_hot_matrix = torch.eye(10)
    for i in range(NUM_EPOCHS):
        model.train()
        average_loss = 0.0
        
        for x_batch, label_batch in dataloader:
            
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            
            output, mu, log_sigma =model(x_batch, one_hot_matrix[label_batch].to(device))
            
            loss = criterion(output, x_batch, mu, log_sigma)
            
            loss.backward()
            optimizer.step()
            
            average_loss += loss
        print(f"Epoch {i+1} completed. Average loss = {average_loss/len(dataloader)}")
  


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    
    # args = parser.parse_args()
    print(device)
    
    transform=v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
    ])
    
    training_set = MNIST(root='./mnist', transform=transform, download=False)
    
    training_loader = DataLoader(training_set, batch_size=128, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True)
    
    model = AutoEncoder()
    model=model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    training_epoch(model, loss_function, optimizer, training_loader)
    
    for i in range(10):
        model.generate_sample(i)
        
    breakpoint()
import torch
import torch.nn as nn
import math
import os
import yaml
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NoiseSchedule:
    def __init__(self, L, s=0.008):
        self.L=L
        t=torch.linspace(0.0, L, L+1, device=device)/L
        a=torch.cos((t+s)/(1+s)*torch.pi/2)**2
        a=a/a[0]
        self.beta=(1-a[1:]/a[:-1]).clip(0.0, 0.99)
        self.alpha=torch.cumprod(1.0-self.beta, dim=0)
        self.one_minus_beta=1-self.beta
        self.one_minus_alpha=1-self.alpha
        self.sqrt_alpha=torch.sqrt(self.alpha)
        self.sqrt_beta=torch.sqrt(self.beta)
        self.sqrt_1_alpha=torch.sqrt(self.one_minus_alpha)
        self.sqrt_1_beta=torch.sqrt(self.one_minus_beta)

    def __len__(self):
        return self.L


class TimeEncoding:
    def __init__(self, L, dim):
        # Note: the dimension dim should be an even number
        self.L=L
        self.dim=dim
        dim2=dim//2
        encoding=torch.zeros(L, dim)
        ang=torch.linspace(0.0, torch.pi/2, L)
        logmul=torch.linspace(0.0, math.log(40), dim2)
        mul=torch.exp(logmul)
        for i in range(dim2):
            a=ang*mul[i]
            encoding[:,2*i]=torch.sin(a)
            encoding[:,2*i+1]=torch.cos(a)
        self.encoding=encoding.to(device=device)

    def __len__(self):
        return self.L

    def __getitem__(self, t):
        return self.encoding[t]


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
                  nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
                  nn.BatchNorm2d(from_features),
                  nn.ReLU(),
                  nn.Conv2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(to_features),
                  nn.ReLU()
          )
          return model

      def build_decoder(self, from_features, to_features):
          model=nn.Sequential(
                  nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
                  nn.BatchNorm2d(from_features),
                  nn.ReLU(),
                  nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(to_features),
                  nn.ReLU()
          )
          return model




class Network(nn.Module):
    def __init__(self, architecture_yaml, train_yaml):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture_yaml=architecture_yaml
        self.train_yaml=train_yaml

        # Extract parameters from the YAML files
        self.num_epochs = self.train_yaml["TRAINING"]["NUM_EPOCHS"]
        self.L = self.train_yaml["TRAINING"]["L"]
        self.cond_shape = self.train_yaml["TRAINING"]["COND_SHAPE"]
        self.time_encodig_size = self.architecture_yaml["TIME_ENCODING_SIZE"]
        self.time_encoding=TimeEncoding(self.L, self.time_encodig_size)
        self.feat_list = self.architecture_yaml["FEAT_LIST"]

        self.pre=nn.Sequential(
            nn.Conv2d(3, self.feat_list[0], 3, padding='same'),
            nn.ReLU())
        self.unet = self.build_unet(64, self.feat_list)
        self.post=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.feat_list[0], 3, 3, padding='same'))
        
        # training parameters
        self.noise_schedule=NoiseSchedule(self.L)
        self.loss_function=nn.MSELoss()
        self.optimizer=torch.optim.AdamW(self.parameters(), lr=self.train_yaml["TRAINING"]["LR"])
        self.image_dimensions=(1,1,1)

        self.cond_one_hot=torch.eye(self.cond_shape[0], device=device)
        self.powers = 2 ** torch.arange(3 - 1, -1, -1).to(self.device)

    def forward(self, x, t, cond):
        enc=self.time_encoding[t]
        x=self.pre(x)
        y=self.unet(x, enc, cond)
        y=self.post(y)
        return y

    def build_unet(self, size, feat_list):
        if len(feat_list)>2:
            inner_block=self.build_unet(size//2, feat_list[1:])
        else:
            inner_block=None
        return UNetBlock(size, self.time_encodig_size, feat_list[0], feat_list[1], self.cond_shape[0], inner_block)

    def training_epoch(self, checkpoint_path, dataloader):
        self.train()
        list_loss = []
        start_epoch=0
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            list_loss = checkpoint['list_loss']
            print(f"Riprendo l'addestramento da epoca {start_epoch+1}...")
        
        for epoch in range(start_epoch, self.num_epochs, 1):
            loss = self.train_one_epoch(dataloader)
            print(f'Epoch: {epoch+1}\n\tLoss: {loss}')
            list_loss.append(loss)
        
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'list_loss': list_loss,
            }
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
        print("FINISH TRANING")


    def train_one_epoch(self, dataloader):
        epoch_count=0
        self.train()
        average_loss=0.0
        for x, y in dataloader:
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
            zt=sqrt_alpha*x + sqrt_1_alpha*eps

            # Compute the output of the network (estimate of eps)
            # and the loss
            g=self(zt,t,cond)
            loss=self.loss_function(g, eps)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            average_loss=0.9*average_loss+0.1*loss.cpu().item()
        epoch_count += 1
        print(f'Epoch {epoch_count} completed. Average loss={average_loss}')
        return average_loss
    
    
    def generate_sample(self, y, save_folder, time, lam=0.5):
        self.eval()
        with torch.no_grad():
            z=torch.randn(1, *(3,64,64), device=device)
            inty = (y.to(device) * self.powers).sum(dim=0).long()
            cond=self.cond_one_hot[inty]
            cond0=torch.zeros_like(cond)
            self.eval()
            for kt in reversed(range(self.L)):
                t=torch.tensor(kt).view(1)

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

        os.makedirs(save_folder, exist_ok=True)
        img = (z.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_folder, f'generated_{time}.png'), img_bgr)
        return z






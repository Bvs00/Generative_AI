import os
import yaml
import torch
import torch.nn as nn
from torchvision.transforms import v2

from VAE import AutoEncoder

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

def training_hp():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load the configuration file
    with open(os.path.join(current_dir, "train_params.yaml"), "r") as f:
        config = yaml.safe_load(f)
    # Access parameters
    MODEL_TYPE = config["MODEL"]["TYPE"]
    MODEL_NAME = config["MODEL"]["NAME"]
    CHECKPOINT_FOLDER = config["MODEL"]["CHECKPOINT_FOLDER"]

    NUM_EPOCHS = config["TRAINING"]["NUM_EPOCHS"]
    LR = config["TRAINING"]["LR"]
    BATCH_SIZE = config["TRAINING"]["BATCH_SIZE"]

    LATENT_SIZE = config["TRAINING"]["LATENT_SIZE"]
    BETA = config["TRAINING"]["BETA"]
    ENCODER_CHANNEL_PROGRESSION = config["TRAINING"]["ENCODER_CHANNEL_PROGRESSION"]
    DECODER_CHANNEL_PROGRESSION = config["TRAINING"]["DECODER_CHANNEL_PROGRESSION"]
    
    # LOSS_BALANCE = config["TRAINING"]["LOSS_BALANCING"]
    WITH_AUGMENTATION = config["TRAINING"]["WITH_AUGMENTATION"]

    OUTPUT_PATH = config["OUTPUT_PATH"]
    


    if WITH_AUGMENTATION:
        print("Using data augmentation")
        custom_transforms=v2.Compose([
            v2.ToImage(),
            v2.CenterCrop((160,160)),
            v2.Resize((64,64)),
            v2.ToDtype(torch.float32, scale=True),
        ])
    else:
        custom_transforms=v2.Compose([
            v2.ToImage(),
            v2.CenterCrop((160,160)),
            v2.Resize((64,64)),
            v2.ToDtype(torch.float32, scale=True),
        ])
        print("Not using data augmentation")

    # Model
    model = AutoEncoder(
        latent_size=LATENT_SIZE,
        encoder_channel_progression=ENCODER_CHANNEL_PROGRESSION,
        decoder_channel_progression=DECODER_CHANNEL_PROGRESSION
    )

    count = 0
    for param in model.parameters(): 
        param.requires_grad = True
        count+=1

    
    criterion = VAELoss(beta=BETA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    checkpoint_path = os.path.join(CHECKPOINT_FOLDER, f"{MODEL_NAME}_{LR}_{BATCH_SIZE}_{LATENT_SIZE}_{BETA}.pth")
    assert not os.path.exists(checkpoint_path), "Already exist a model with this configuration, please change the parameters in train_params.yaml"

    return MODEL_NAME, MODEL_TYPE, checkpoint_path, model, optimizer, criterion, BATCH_SIZE, NUM_EPOCHS, OUTPUT_PATH, custom_transforms
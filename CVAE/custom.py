import os
import yaml
import torch
import torch.nn as nn
from torchvision.transforms import v2
import VAE

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
    RESUME_FROM_CHECKPOINT = config["MODEL"]["RESUME_FROM_CHECKPOINT"]

    NUM_EPOCHS = config["TRAINING"]["NUM_EPOCHS"]
    LR = config["TRAINING"]["LR"]
    BATCH_SIZE = config["TRAINING"]["BATCH_SIZE"]
    BETA = config["TRAINING"]["BETA"]

    ARCHITECTURE_YAML_NAME = config["MODEL"]["ARCHITECTURE_YAML_NAME"]
    
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

    # Load the configuration file
    with open(os.path.join(current_dir, "architectures_yaml", ARCHITECTURE_YAML_NAME), "r") as f:
        arch_config = yaml.safe_load(f)
    LATENT_SIZE = arch_config["LATENT_SIZE"]

    # Model instance
    model_class = getattr(VAE, arch_config["CLASS_NAME"])
    model = model_class(arch_config)

    for param in model.parameters(): 
        param.requires_grad = True
    
    criterion = VAELoss(beta=BETA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    checkpoint_path = os.path.join(CHECKPOINT_FOLDER, f"{MODEL_NAME}_{LR}_{BATCH_SIZE}_{LATENT_SIZE}_{BETA}.pth")
    if os.path.exists(checkpoint_path) and RESUME_FROM_CHECKPOINT:
        print(f"Resuming from checkpoint: {checkpoint_path}")
    elif not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found, will create a new one: {checkpoint_path}")
    else:
        assert False, "Checkpoint exists but RESUME_FROM_CHECKPOINT is set to False, please check your configuration."

    return MODEL_NAME, MODEL_TYPE, checkpoint_path, model, optimizer, criterion, BATCH_SIZE, NUM_EPOCHS, OUTPUT_PATH, custom_transforms


def test_hp():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load the configuration file
    with open(os.path.join(current_dir, "train_params.yaml"), "r") as f:
        config = yaml.safe_load(f)

    ARCHITECTURE_YAML_NAME = config["MODEL"]["ARCHITECTURE_YAML_NAME"]
    
    # Load the configuration file
    with open(os.path.join(current_dir, "architectures_yaml", ARCHITECTURE_YAML_NAME), "r") as f:
        arch_config = yaml.safe_load(f)

    # Model instance
    model_class = getattr(VAE, arch_config["CLASS_NAME"])
    model = model_class(arch_config)

    for param in model.parameters(): 
        param.requires_grad = False

    return model
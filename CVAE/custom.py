import os
import yaml
import torch
import torch.nn as nn
from torchvision.transforms import v2
import VAE


def training_hp():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load the configuration file
    with open(os.path.join(current_dir, "train_params.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Access parameters
    MODEL_NAME = config["MODEL"]["NAME"]
    CHECKPOINT_FOLDER = config["MODEL"]["CHECKPOINT_FOLDER"]
    RESUME_FROM_CHECKPOINT = config["MODEL"]["RESUME_FROM_CHECKPOINT"]

    LR = config["TRAINING"]["LR"]
    NUM_WORKERS = config["TRAINING"]["NUM_WORKERS"]
    BATCH_SIZE = config["TRAINING"]["BATCH_SIZE"]
    BETA = config["TRAINING"]["BETA"]

    ARCHITECTURE_YAML_NAME = config["MODEL"]["ARCHITECTURE_YAML_NAME"]
    
    # LOSS_BALANCE = config["TRAINING"]["LOSS_BALANCING"]
    WITH_AUGMENTATION = config["TRAINING"]["WITH_AUGMENTATION"]
    
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

    # Model instance
    model_class = getattr(VAE, arch_config["CLASS_NAME"])
    model = model_class(arch_config, config)

    for param in model.parameters(): 
        param.requires_grad = True

    LATENT_SIZE = arch_config["LATENT_SIZE"]
    checkpoint_path = os.path.join(CHECKPOINT_FOLDER, f"{MODEL_NAME}_{LR}_{BATCH_SIZE}_{LATENT_SIZE}_{BETA}.pth")
    if os.path.exists(checkpoint_path) and RESUME_FROM_CHECKPOINT:
        print(f"Resuming from checkpoint: {checkpoint_path}")
    elif not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found, will create a new one: {checkpoint_path}")
    else:
        assert False, "Checkpoint exists but RESUME_FROM_CHECKPOINT is set to False, please check your configuration."

    return checkpoint_path, model, BATCH_SIZE, NUM_WORKERS, custom_transforms


def test_hp():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load the configuration file
    with open(os.path.join(current_dir, "train_params.yaml"), "r") as f:
        config = yaml.safe_load(f)

    MODEL_TYPE = config["MODEL"]["TYPE"]
    ARCHITECTURE_YAML_NAME = config["MODEL"]["ARCHITECTURE_YAML_NAME"]
    
    # Load the configuration file
    with open(os.path.join(current_dir, "architectures_yaml", ARCHITECTURE_YAML_NAME), "r") as f:
        arch_config = yaml.safe_load(f)

    # Model instance
    model_class = getattr(VAE, arch_config["CLASS_NAME"])
    model = model_class(arch_config, config)

    for param in model.parameters(): 
        param.requires_grad = False

    return model, MODEL_TYPE
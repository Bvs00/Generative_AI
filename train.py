import os
os.environ["MPLCONFIGDIR"] = os.path.expanduser("/tmp/matplotlib")
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CelebA
from matplotlib import pyplot as plt
import argparse
from CelebA import CelebADataset
import random
import numpy as np
import yaml
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--relative_path", type=str, required=True, 
                        help="Relative path to the dataset")
    args = parser.parse_args()


    sys.path.insert(1, args.relative_path)
    from custom import training_hp

    # Load the configuration file
    with open(os.path.join(args.relative_path,"train_params.yaml"), "r") as f:
        config = yaml.safe_load(f)

    TEST_ONLY = config["TRAINING"]["TEST_ONLY"]
    DATASET_PATH = config["DATASET_PATH"]


    random.seed(1)  # Set seed for Python's random
    np.random.seed(1)  # Set seed for NumPy
    torch.manual_seed(1)  # Set seed for PyTorch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path, model, batch_size, custom_transforms = training_hp()
    
    model=model.to(device)

    # Count trainable and non-trainable parameters
    trainable_params = torch.sum(torch.tensor(list(p.numel() for p in model.parameters() if p.requires_grad))).item()
    non_trainable_params = torch.sum(torch.tensor(list(p.numel() for p in model.parameters() if not p.requires_grad))).item()

    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    print(f"Total parameters: {trainable_params + non_trainable_params}")

    # training_set = CelebA(root='./celeba', transform=transform, download=False, split='train')
    dataset = CelebA(root=DATASET_PATH, split='all', transform=custom_transforms, target_type="attr", download=False)
    
    training_loader = DataLoader(CelebADataset(dataset), batch_size=128, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True)

    model.training_epoch(checkpoint_path, training_loader)
    
    

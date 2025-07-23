import os
import yaml
import torch
import torch.nn as nn
from torchvision.transforms import v2
import GUIDED_DIFFUSION
from GUIDED_DIFFUSION import find_matching_checkpoint


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
    L = config["TRAINING"]["L"]
    BATCH_SIZE = config["TRAINING"]["BATCH_SIZE"]
    NUM_WORKERS = config["TRAINING"]["NUM_WORKERS"]

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
    model_class = getattr(GUIDED_DIFFUSION, arch_config["COMMON_CLASS_NAME"])
    model = model_class(arch_config, config)
    model.set_trining_parameters()
    time_encoding_size = arch_config["TIME_ENCODING_SIZE"]
    feat_list = arch_config["FEAT_LIST"]

    for param in model.parameters(): 
        param.requires_grad = True
    
    scheduleType = config["TRAINING"].get("SCHEDULE_TYPE", "")

    # Build the checkpoint filename as before
    checkpoint_filename = f"{MODEL_NAME}_archname_{arch_config['COMMON_CLASS_NAME']}_lr_{LR}_{BATCH_SIZE}_{time_encoding_size}_{feat_list}_{L}_{scheduleType}"
    checkpoint_path = os.path.join(CHECKPOINT_FOLDER, checkpoint_filename + ".pth")

    # Check for existence
    existing_checkpoint = None
    if RESUME_FROM_CHECKPOINT:
        existing_checkpoint = find_matching_checkpoint(CHECKPOINT_FOLDER, checkpoint_filename)
        if existing_checkpoint:
            print(f"Resuming from checkpoint: {existing_checkpoint}")
            # Example load:
            # model.load_state_dict(torch.load(existing_checkpoint))
        else:
            print(f"Checkpoint not found, will create a new one: {checkpoint_path}")
    if not RESUME_FROM_CHECKPOINT and existing_checkpoint:
        assert False, "Checkpoint exists but RESUME_FROM_CHECKPOINT is set to False, please check your configuration."

    return checkpoint_path, model, BATCH_SIZE, NUM_WORKERS, custom_transforms


def test_hp(cp_path):
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
    model_class = getattr(GUIDED_DIFFUSION, arch_config["COMMON_CLASS_NAME"])
    model = model_class(arch_config, config)
    model.set_trining_parameters()

    for param in model.parameters(): 
        param.requires_grad = False
    
    model.load_state_dict(torch.load(cp_path)['model_state_dict'])

    return model, MODEL_TYPE
import argparse
from itertools import product
import os
import yaml
import torch
import sys
from colorama import Fore, Style, init

# python image_generation.py --relative_path ../Generative_AI/GUIDED-DIFFUSION --cp_path "/user/mmarseglia/GenAI/Generative_AI/GUIDED-DIFFUSION/checkpoints/baseline_archname_Network_lr_0.001_128_64_[32, 64, 128, 256]_1000.pth" --save_folder ./Image_Generated/GUIDED-DIFFUSION --sample_per_class 20
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--relative_path", type=str, required=True, 
                        help="Relative path to the dataset")
    parser.add_argument("--cp_path", type=str, required=True,
                        help="Path to the model checkpoint file")
    parser.add_argument("--save_folder", type=str, default="./Image_Generated/VAE",
                        help="Path to save the generated samples")
    parser.add_argument("--sample_per_class", "-s", type=int, default=20,
                        help="Number of samples to generate per class")
    args = parser.parse_args()

    init(autoreset=True) # Initialize colorama for colored output

    sys.path.insert(1, args.relative_path)
    from custom import test_hp

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(Fore.YELLOW + "Loading model...")
    model, model_type = test_hp(args.cp_path)

    # model.load_state_dict(torch.load(args.cp_path)['model_state_dict'])
    model.to(device)
    
    print("device: " + device)
    #print colored text
    print(Fore.GREEN + "Model loaded successfully")
    print("evaluating: " + model_type)
    print(Fore.YELLOW + "Generating samples...")
    if model_type == "DModel":
        lambda_values = [0.5, 1, 2, 3]

    # for bits in product([0, 1], repeat=3):
    #     y = torch.tensor(bits, dtype=torch.float)
    #     label_str = '_'.join(str(int(v)) for v in y.tolist())
    #     for time in range(args.sample_per_class):
    #         if model_type == "DModel":
    #             for lambda_v in lambda_values:
    #                 lambda_save_folder = f"{args.save_folder}/lambda_{lambda_v}"
    #                 os.makedirs(args.save_folder, exist_ok=True)
    #                 model.generate_sample(y, save_folder=f"{lambda_save_folder}/{label_str}", time=time, lam=lambda_v)
    #         else:
    #             os.makedirs(args.save_folder, exist_ok=True)
    #             model.generate_sample(y, save_folder=f"{args.save_folder}/{label_str}", time=time)
    # print(Fore.GREEN + "Samples generated successfully")


    # Prepare full batch of conditions and corresponding save paths
    condition_list = []
    save_paths = []

    bit_combinations = list(product([0, 1], repeat=3))

    for bits in bit_combinations:
        y = torch.tensor(bits, dtype=torch.float)
        label_str = '_'.join(str(int(v)) for v in y.tolist())

        for time in range(args.sample_per_class):
            condition_list.append(y)
            save_paths.append(f"{args.save_folder}/{label_str}")

    # Stack into full batch tensor
    y_batch = torch.stack(condition_list)  # Shape: (8 * samples_per_class, 3)

    if model_type == "DModel":
        for lambda_v in lambda_values:
            # Optionally modify the save path to include lambda subfolder
            lambda_save_paths = [
                path.replace(args.save_folder, os.path.join(str(args.save_folder), f"lambda_{lambda_v}"))
                for path in save_paths
            ]
            for folder in lambda_save_paths:
                os.makedirs(folder, exist_ok=True)
            print(Fore.YELLOW + f"Generating samples with lambda = {lambda_v}...")
            model.generate_sample(y_batch, save_folders=lambda_save_paths, lam=lambda_v)
    else:
        # Create folders
        for path in save_paths:
            # Ensure the save path exists
            os.makedirs(path, exist_ok=True)
        model.generate_sample(y_batch, save_folders=save_paths)

    print(Fore.GREEN + "Samples generated successfully")
import argparse
from itertools import product
import os
import yaml
import torch
import sys
from colorama import Fore, Style, init


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
    model = test_hp(args.cp_path)

    # model.load_state_dict(torch.load(args.cp_path)['model_state_dict'])
    model.to(device)

    #print colored text
    print(Fore.GREEN + "Model loaded successfully")
    print(Fore.YELLOW + "Generating samples...")
    for bits in product([0, 1], repeat=3):
        y = torch.tensor(bits, dtype=torch.float)
        label_str = '_'.join(str(int(v)) for v in y.tolist())
        for time in range(args.sample_per_class):
            os.makedirs(args.save_folder, exist_ok=True)
            model.generate_sample(y, save_folder=f"{args.save_folder}/{label_str}", time=time)
    print(Fore.GREEN + "Samples generated successfully")
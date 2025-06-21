import os
os.environ["MPLCONFIGDIR"] = os.path.expanduser("/tmp/matplotlib")
import torch
import argparse
from Generative_AI.CVAE.VAE import AutoEncoder
from itertools import product


factory_method_model = {
    'VAE': AutoEncoder
}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_path', type=str, default="Image_Generated/VAE")
    parser.add_argument('--name_model', type=str, default="Results_VAE_CelebA/VAE")
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--label', nargs="+", required=True)
    args = parser.parse_args()
    
    name_model = args.name_model.split('/')[-1]
    
    model = factory_method_model[name_model](args=args)
    model.load_state_dict(torch.load(f"{args.name_model}_{args.latent_size}_checkpoint.pth")['model_state_dict'])
    model = model.to(device=args.device)
    
    y = torch.tensor([int(x) for x in args.label], dtype=torch.float, device=args.device)
    
    for bits in product([0, 1], repeat=3):
        y = torch.tensor(bits)
        model.generate_sample(y, path=args.saved_path)
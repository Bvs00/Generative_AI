import torch

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        attr_names = base_dataset.attr_names
        
        self.male_idx = attr_names.index("Male")
        self.eyeglasses_idx = attr_names.index("Eyeglasses")
        self.beard_idx = [attr_names.index("No_Beard"), attr_names.index("Goatee"), attr_names.index("Mustache")]
    
    def extract_condition(self, attr):
        gender = attr[self.male_idx]
        glasses = attr[self.eyeglasses_idx]
        beard = 1 - attr[self.beard_idx[0]] or attr[self.beard_idx[1]] or attr[self.beard_idx[2]]  # True if has any beard
        return torch.tensor([gender, beard, glasses], dtype=torch.float)

    def __getitem__(self, index):
        img, attr = self.base[index]
        cond = self.extract_condition(attr)
        return img, cond

    def __len__(self):
        return len(self.base)
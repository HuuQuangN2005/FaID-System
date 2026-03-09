import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CelebALandmarkDataset(Dataset):

    def __init__(self, img_dir, split_file):

        self.img_dir = img_dir
        self.samples = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            start_idx = 1 if "lefteye_x" in lines[0] else 0

            for line in lines[start_idx:]:

                parts = line.strip().split()

                img_name = parts[0]
                points = list(map(float, parts[1:11]))

                self.samples.append((img_name, points))

        print(f"Loaded {len(self.samples)}")

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_name, points = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        landmarks = torch.tensor(points).float().view(5,2)

        landmarks[:,0] /= 178.0
        landmarks[:,1] /= 218.0

        return image, landmarks
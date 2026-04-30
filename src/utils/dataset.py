import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2


class WebFaceDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label"].tolist()
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.filepaths[idx])
        image = Image.open(img_path).convert("RGB")
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


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

        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, points = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        landmarks = torch.tensor(points).float().view(5, 2)

        landmarks[:, 0] /= 178.0
        landmarks[:, 1] /= 218.0

        return image, landmarks


class RecognitionDataset(Dataset):
    def __init__(self, data_folder: str, images: list, labels: list, transform=None):
        if not os.path.exists(data_folder):
            raise ValueError("path not exist!!")

        self.data_folder = data_folder
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        class_name: str = filename.split("_")[0]

        image = Image.open(os.path.join(self.data_folder, filename)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(
            self.labels.index(class_name.lower()), dtype=torch.long
        )

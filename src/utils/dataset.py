from torch.utils.data import  Dataset
from PIL import Image
import os 
import torch

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

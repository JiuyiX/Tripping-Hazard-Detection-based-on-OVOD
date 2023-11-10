from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class ExpDataset(Dataset):

    def __init__(self, root_dir, label_dir, scene_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.scene_dir = scene_dir
        self.path = os.path.join(self.root_dir, self.label_dir, self.scene_dir)
        self.img_path = [img_name for img_name in os.listdir(self.path) if (not img_name.endswith('aug1.jpg') and (not img_name.endswith('aug2.jpg')))]

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, self.scene_dir, img_name)
        img = Image.open(img_item_path)

        normalized_tensor = transform(img)

        if self.label_dir == "hazard":
            label = 1
        else:
            label = 0

        return normalized_tensor, label

    def __len__(self):
        return len(self.img_path)

class ExpDataset_aug(Dataset):

    def __init__(self, root_dir, label_dir, scene_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.scene_dir = scene_dir
        self.path = os.path.join(self.root_dir, self.label_dir, self.scene_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, self.scene_dir, img_name)
        img = Image.open(img_item_path)

        normalized_tensor = transform(img)
        
        if self.label_dir == "hazard":
            label = 1
        else:
            label = 0

        return normalized_tensor, label

    def __len__(self):
        return len(self.img_path)
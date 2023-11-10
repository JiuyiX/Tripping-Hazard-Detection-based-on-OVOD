from PIL import Image
from torchvision import transforms
import glob
import random

def data_augmentation1(img_path):
    image = Image.open(img_path)
    augmented_image = image.copy()

    color_jitter = transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
    )

    augmented_image = color_jitter(augmented_image)

    if random.random() < 0.5:
        augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
    
    augmented_image.save(f"{img_path[:-4]}_aug1.jpg")

def data_augmentation2(img_path):
    image = Image.open(img_path)
    augmented_image = image.copy()

    color_jitter = transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
    )

    augmented_image = color_jitter(augmented_image)

    if random.random() < 0.5:
        augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
    
    augmented_image.save(f"{img_path[:-4]}_aug2.jpg")

for img_path in glob.glob(r"TH_in_outdoor_source/*/*/*/*.jpg"):
    print(img_path)
    data_augmentation1(img_path)
    data_augmentation2(img_path)
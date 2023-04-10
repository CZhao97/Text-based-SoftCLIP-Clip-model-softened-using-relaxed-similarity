from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import pylab

# Define dataset
class COCODataset(Dataset):
    def __init__(self, coco, img_dir, text_model, transform = None):
        self.coco = coco
        self.img_dir = img_dir
        self.ids = list(coco.anns.keys())
        self.transform = transform
        self.text_model = text_model

    def __getitem__(self, index):
        annotation_id = self.ids[index]
        annotation = self.coco.anns[annotation_id]
        text = annotation['caption']

        img_id = annotation["image_id"]
        img_name = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, text

    def __len__(self):
        return len(self.ids)

# Loading data
def load_as_dataset(dataType, batch_size,dir = 'coco', trans_type = None, text_model = None):
    img_dir = '{}/{}2017'.format(dir, dataType)
    annFile='{}/annotations/captions_{}2017.json'.format(dir, dataType)
    coco=COCO(annFile)

    # Define transform function
    if trans_type == None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to a fixed size
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
    
    # The simplify version adopted by original CLIP
    elif trans_type == 'random resize crop':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]) 

    
    # Create dataset
    coco_dataset = COCODataset(coco, img_dir, text_model, transform)

    # Load as batches
    coco_dataloader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return coco_dataloader


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
    def __init__(self, coco, img_dir, text_model, detector, transform = None, process="train"):
        self.coco = coco
        self.img_dir = img_dir
        self.ids = list(coco.anns.keys())
        self.transform = transform
        self.text_model = text_model
        self.detector = detector
        self.detector_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __getitem__(self, index):
        annotation_id = self.ids[index]
        annotation = self.coco.anns[annotation_id]
        text = annotation['caption']
        img_id = annotation["image_id"]
        img = self.coco.loadImgs(img_id)[0]
        img_name = img['file_name']

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = np.asarray(image)

        # boxes, scores, class_ids
        if process!="train":
            _, _, class_ids = self.detector(image)
            label = [self.detector_classes[class_id] for class_id in class_ids]
            cat_sentence = ', '.join(np.unique(np.array(label)))
            
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        if process!="train":
            return image, text
        else:
            return image, text, cat_sentence
    

    def __len__(self):
        return len(self.ids)

# Loading data
def load_as_dataset(dataType, batch_size, detector, dir = 'coco', trans_type = None, text_model = None, process="train"):
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
    coco_dataset = COCODataset(coco, img_dir, text_model, detector, transform, process)

    # Load as batches
    coco_dataloader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=True)

    return coco_dataloader


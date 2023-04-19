import torch
import torch.nn.functional as F
from Model import Pclip, nt_xent_loss
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, ViTFeatureExtractor, ViTModel
from types import SimpleNamespace
import torchvision.transforms as transforms
from Load_data import load_as_dataset
import argparse, os
import gc
import numpy as np
import tracemalloc
from tqdm import tqdm
import cv2
from imread_from_url import imread_from_url
import sys
sys.path.insert(0, 'yolo_v7/ONNX-YOLOv7-Object-Detection')
from yolov7 import YOLOv7



def train_and_test(args):
    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    dir = args.dir
    batch_size = args.batch_size
    config = {
        'text_model': args.text_model,
        'img_model': args.img_model,
        'embedding_size': args.embedding_size,
        'similarity_method': args.similarity_method,
        'dropout': args.dropout,
        'process': "train"
    }
    config = SimpleNamespace(**config)

    yolov7_detector = YOLOv7(args.yolo_weight_path, conf_thres=args.yolo_conf_thres, iou_thres=args.yolo_iou_thres)

    model = Pclip(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trans_type = 'random resize crop'
    to_pil_image = transforms.ToPILImage()
    
    dataType = 'train'
    train_dataset = load_as_dataset(dataType, batch_size, yolov7_detector, dir, trans_type, args.text_model, "train")
    model.train()

    if args.text_model == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.text_model == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    if args.img_model == 'ViT-L/14':
        processor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224")
    elif args.img_model == 'ViT-B/32':
        processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    img_is_freeze = False
    text_is_freeze = False
    for epoch in range(args.epochs):
        train_loss = 0
        num_batches = 0
        loss_100 = 0

        for img, text, cat_sentence in tqdm(train_dataset):
            if num_batches > args.freeze_iteration:
                if (not text_is_freeze) and args.freeze_text:
                    for param in model.text_encoder.parameters():
                        param.requires_grad = False
                        text_is_freeze = True
                if (not img_is_freeze) and args.freeze_img:
                    for param in model.img_encoder.parameters():
                        param.requires_grad = False
                        img_is_freeze = True
            

            caption = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            ids, masks = caption['input_ids'], caption['attention_mask']

            tokenized_cat = tokenizer(cat_sentence, padding=True, truncation=True, return_tensors="pt")
            cat_ids, cat_masks = tokenized_cat['input_ids'], tokenized_cat['attention_mask']

            if args.img_model != 'resnet50':
                img = [to_pil_image(image) for image in img]
                img = processor(img, return_tensors="pt")

            img, ids, masks, cat_ids, cat_masks = img.to(device), ids.to(device), masks.to(device), cat_ids.to(device), cat_masks.to(device)
            similarity_matrix, similarity_matrix_text = model(img, ids, masks, cat_ids, cat_masks)
            optimizer.zero_grad()
            amount = similarity_matrix.size(0)

            # For step2, we can change the labels here by our new designed soft labels based on object classes.
            if args.loss_function == 'hard-NT-Xent':
                labels = torch.eye(amount).to(device)
                loss = nt_xent_loss(similarity_matrix, labels)

            if args.loss_function == 'soft-NT-Xent':
                labels = 0.7*torch.eye(amount).to(device) + 0.3*similarity_matrix_text
                loss = nt_xent_loss(similarity_matrix, labels)

            loss.backward()
            optimizer.step()
            
            num_batches += 1
            train_loss += loss.item()
            loss_100 += loss.item()

            del img, ids, masks, similarity_matrix, labels, loss, amount, caption, text, cat_ids, cat_masks

            if num_batches % 100 == 0:
                allocated_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)
                reserved_memory = torch.cuda.memory_reserved(device) / (1024 * 1024)

                print('\nThe average loss of {} to {} iteration is: '.format(num_batches - 99, num_batches), loss_100 / 100, flush=True)
                print(f"Allocated GPU memory: {allocated_memory:.2f} MB", flush=True)
                print(f"Reserved GPU memory: {reserved_memory:.2f} MB", flush=True)
                
                loss_100 = 0

            if num_batches % 1000 == 0:
                torch.cuda.empty_cache()
                collected = gc.collect()
                print('Cache cleaned.')
                print(f"Garbage collector collected {collected} objects.")
            
 
        train_loss = train_loss / num_batches
        print(f'\nTraining loss for epoch {epoch + 1}: {train_loss : .3f}', flush=True)
    

    
    if args.save_model:
        torch.save(model.state_dict(), args.save_dir)
    
    dataType = 'val'
    accuracy_matrix_size = 128
    test_dataset = load_as_dataset(dataType, accuracy_matrix_size,yolov7_detector, dir, trans_type, args.text_model, "test")
    
    model.eval()

    model.process="test"

    with torch.no_grad():
        totalAccuracy = 0
        num_test = 0
        for img, text in tqdm(test_dataset):
            caption = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            ids, masks = caption['input_ids'], caption['attention_mask']

            if args.img_model != 'resnet50':
                to_pil_image = transforms.ToPILImage()
                images = [to_pil_image(image) for image in img]
                img = processor(images, return_tensors="pt")

            img, ids, masks = img.to(device), ids.to(device), masks.to(device)
            similarity_matrix = model(img, ids, masks, None, None)


            pred_label = torch.argmax(similarity_matrix, dim=0, keepdim=False)
            if args.gpu:
                pred_label = pred_label.cpu()
            pred_label = pred_label.numpy()
            labels = np.array(range(accuracy_matrix_size))
            accuracy = np.sum(labels == pred_label) / accuracy_matrix_size

            num_test += 1
            totalAccuracy += accuracy
    
    totalAccuracy /= num_test
    print(f'\nThe test accuracy is: {totalAccuracy}', flush=True)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="coco")
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--save_dir", type=str, default="models/model.pt")


    # hyper parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--embedding_size", type=int, default=768)
    
    # hyper parameters we often adjust
    parser.add_argument("--freeze_text", action='store_true')
    parser.add_argument("--freeze_img", action='store_true')
    parser.add_argument("--freeze_iteration", type=int, default=1000)
    parser.add_argument("--img_model", type=str, default='ViT-B/32')
    parser.add_argument("--text_model", type=str, default='bert-base-uncased')
    parser.add_argument("--similarity_method", type=str, default='cos_similarity')
    parser.add_argument("--loss_function", type=str, default='soft-NT-Xent')
    parser.add_argument("--yolo_weight_path", type=str, default="yolo_v7/ONNX-YOLOv7-Object-Detection/models/yolov7_640x640.onnx")
    parser.add_argument("--yolo_conf_thres", type=float, default=0.2)
    parser.add_argument("--yolo_iou_thres", type=float, default=0.3)

    args = parser.parse_args()
    print(f"args: {vars(args)}", flush=True)
    return args

if __name__ == "__main__":
    args = get_args()
    train_and_test(args)
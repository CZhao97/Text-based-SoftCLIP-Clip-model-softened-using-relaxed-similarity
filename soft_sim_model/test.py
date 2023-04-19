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

def test(args):
    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    dir = args.dir
    batch_size = args.batch_size
    config = {
        'text_model': args.text_model,
        'img_model': args.img_model,
        'embedding_size': args.embedding_size,
        'similarity_method': args.similarity_method,
        'dropout': args.dropout,
        'process': "test"
    }
    config = SimpleNamespace(**config)


    model = Pclip(config)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_dir))
    trans_type = 'random resize crop'
    to_pil_image = transforms.ToPILImage()

    if args.text_model == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.text_model == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    if args.img_model == 'ViT-L/14':
        processor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224")
    elif args.img_model == 'ViT-B/32':
        processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")


    dataType = 'val'
    accuracy_matrix_size = 128
    test_dataset = load_as_dataset(dataType, accuracy_matrix_size, yolov7_detector, dir, trans_type, args.text_model, "test")
    model.eval()

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
    parser.add_argument("--model_dir", type=str, default="models/model.pt")

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
    parser.add_argument("--loss_function", type=str, default='hard-NT-Xent')
    parser.add_argument("--yolo_weight_path", type=str, default="yolo_v7/ONNX-YOLOv7-Object-Detection/models/yolov7_640x640.onnx")
    parser.add_argument("--yolo_conf_thres", type=float, default=0.2)
    parser.add_argument("--yolo_iou_thres", type=float, default=0.3)


    args = parser.parse_args()
    print(f"args: {vars(args)}", flush=True)
    return args

if __name__ == "__main__":
    args = get_args()
    test(args)
import torch
import torch.nn.functional as F
from Model import Pclip, nt_xent_loss
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, ViTFeatureExtractor, ViTModel
from types import SimpleNamespace
import torchvision.transforms as transforms
from Load_data import load_as_dataset
import argparse, os
import numpy as np
from tqdm import tqdm

def train_and_test(args):
    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    dir = args.dir
    batch_size = args.batch_size
    config = {
        'text_model': args.text_model,
        'img_model': args.img_model,
        'embedding_size': args.embedding_size,
        'similarity_method': args.similarity_method,
        'dropout': args.dropout
    }
    config = SimpleNamespace(**config)

    model = Pclip(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trans_type = 'random resize crop'
    
    dataType = 'train'
    train_dataset = load_as_dataset(dataType, batch_size, dir, trans_type, args.text_model)
    model.train()

    if args.text_model == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.text_model == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    if args.img_model == 'ViT-L/14':
        processor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224")
    elif args.img_model == 'ViT-B/32':
        processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    for epoch in range(args.epochs):
        train_loss = 0
        num_batches = 0
        loss_1000 = 0

        for img, text in tqdm(train_dataset):
            caption = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            ids, masks = caption['input_ids'], caption['attention_mask']

            if args.img_model != 'resnet50':
                to_pil_image = transforms.ToPILImage()
                images = [to_pil_image(image) for image in img]
                img = processor(images, return_tensors="pt")

            img, ids, masks = img.to(device), ids.to(device), masks.to(device)
            similarity_matrix = model(img, ids, masks)
            optimizer.zero_grad()
            amount = similarity_matrix.size(0)

            # For step2, we can change the labels here by our new designed soft labels based on object classes.
            if args.loss_function == 'hard-NT-Xent':
                labels = torch.eye(amount).to(device)
                loss = nt_xent_loss(similarity_matrix, labels)

            loss.backward()
            optimizer.step()
            
            num_batches += 1
            train_loss += loss
            loss_1000 += loss

            if num_batches % 1000 == 0:
                print('The average loss of {} to {} iteration is: '.format(num_batches - 999, num_batches), loss_100 / 1000)
                loss_1000 = 0
            
 
        train_loss = train_loss / num_batches
        print(f'Training loss for epoch {epoch + 1}: {train_loss : .3f}')
    
        dataType = 'val'
        accuracy_matrix_size = 128
        train_dataset = load_as_dataset(dataType, accuracy_matrix_size, dir, trans_type, args.text_model)
        model.eval()

        with torch.no_grad():
            totalAccuracy = 0
            num_test = 0
            for img, text in tqdm(train_dataset):
                caption = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
                ids, masks = caption['input_ids'], caption['attention_mask']

                if args.img_model != 'resnet50':
                    to_pil_image = transforms.ToPILImage()
                    images = [to_pil_image(image) for image in img]
                    img = processor(images, return_tensors="pt")

                img, ids, masks = img.to(device), ids.to(device), masks.to(device)
                similarity_matrix = model(img, ids, masks)

                pred_label = torch.argmax(similarity_matrix, dim=0, keepdim=False)
                if args.gpu:
                    pred_label = pred_label.cpu()
                pred_label = pred_label.numpy()
                labels = np.array(range(accuracy_matrix_size))
                accuracy = np.sum(labels == pred_label) / accuracy_matrix_size

                num_test += 1
                totalAccuracy += accuracy
        
        totalAccuracy /= num_test
        print('The test accuracy is: {totalAccuracy}')

                
    
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="coco")
    parser.add_argument("--gpu", action='store_true')

    # hyper parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--embedding_size", type=int, default=768)
    
    # hyper parameters we often adjust
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--img_model", type=str, default='ViT-B/32')
    parser.add_argument("--text_model", type=str, default='bert-base-uncased')
    parser.add_argument("--similarity_method", type=str, default='cos_similarity')
    parser.add_argument("--loss_function", type=str, default='hard-NT-Xent')


    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    train_and_test(args)
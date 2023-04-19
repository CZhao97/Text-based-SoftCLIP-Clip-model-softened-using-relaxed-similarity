import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, ViTFeatureExtractor, ViTModel
import torchvision
import torch.nn.functional as F


class Pclip(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = config.text_model
        self.img_model = config.img_model
        self.similarity_method = config.similarity_method
        self.embedding_size = config.embedding_size
        self.dropout = config.dropout
        self.process = config.process

        # For text encoder
        if self.text_model == 'bert-base-uncased':
            #self.tokenizer = BertTokenizer.from_pretrained(self.text_model)
            self.text_encoder = BertModel.from_pretrained(self.text_model)
            self.text_linear = nn.Linear(self.text_encoder.config.hidden_size, self.embedding_size)
            self.cat_sentence_text_encoder = BertModel.from_pretrained(self.text_model)
            for name, param in self.cat_sentence_text_encoder.named_parameters():
                param.requires_grad = False
        
        if self.text_model == 'roberta-base':
            #self.tokenizer = RobertaTokenizer.from_pretrained(self.text_model)
            self.text_encoder = RobertaModel.from_pretrained(self.text_model)
            self.text_linear = nn.Linear(self.config.hidden_size, self.embedding_size)
            self.cat_sentence_text_encoder = RobertaModel.from_pretrained(self.text_model)
            for name, param in self.cat_sentence_text_encoder.named_parameters():
                param.requires_grad = False
        
        # For image encoder
        if self.img_model == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
            self.img_encoder = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.img_linear = nn.Linear(2048, self.embedding_size)
        
        elif self.img_model == 'ViT-B/32':
            vit_model_name = "google/vit-base-patch16-224"
            #self.processor = ViTFeatureExtractor.from_pretrained(vit_model_name)
            self.img_encoder = ViTModel.from_pretrained(vit_model_name)
            self.img_linear = nn.Linear(768, self.embedding_size)
        
        elif self.img_model == 'ViT-L/14':
            vit_model_name = "google/vit-large-patch16-224"
            #self.processor = ViTFeatureExtractor.from_pretrained(vit_model_name)
            self.img_encoder = ViTModel.from_pretrained(vit_model_name)
            self.img_linear = nn.Linear(768, self.embedding_size)
        
    def forward(self, img, ids, masks, cat_ids, cat_masks):
        if self.img_model == 'resnet50':
            img_embed = self.img_encoder(img)
            img_embed = img_embed.view(img_embed.size(0), -1)
        elif self.img_model == 'ViT-B/32' or self.img_model == 'ViT-L/14':
            #input_img = self.processor(images=img, return_tensors="pt")
            img_embed = self.img_encoder(**img).last_hidden_state[:, 0, :]

        img_embed = self.img_linear(img_embed)
        img_embed = img_embed / torch.norm(img_embed, dim=1, keepdim=True)

        if self.process == "train":

            if self.text_model == 'bert-base-uncased':
                text_embed = self.text_encoder(input_ids=ids, attention_mask=masks).pooler_output
                text_embed_cat = self.cat_sentence_text_encoder(input_ids=cat_ids, attention_mask=cat_masks).pooler_output
                text_embed_cap = self.cat_sentence_text_encoder(input_ids=ids, attention_mask=masks).pooler_output
            elif self.text_model == 'roberta-base':
                text_embed = self.text_encoder(input_ids=ids, attention_mask=masks)[:, 0, :]
                text_embed_cat = self.cat_sentence_text_encoder(input_ids=cat_ids, attention_mask=cat_masks)[:, 0, :]
                text_embed_cap = self.cat_sentence_text_encoder(input_ids=ids, attention_mask=masks)[:, 0, :]
                
            text_embed = self.text_linear(text_embed)
            text_embed = text_embed / torch.norm(text_embed, dim=1, keepdim=True)
            
            text_embed_cat = text_embed_cat / torch.norm(text_embed_cat, dim=1, keepdim=True)
            text_embed_cap = text_embed_cap / torch.norm(text_embed_cap, dim=1, keepdim=True)

            # Compare similarity
            if self.similarity_method == 'cos_similarity':
                similarity_matrix = (torch.matmul(img_embed, text_embed.T) + torch.matmul(text_embed, img_embed.T)) / 2
                similarity_matrix_text = ((torch.matmul(text_embed_cat, text_embed_cap.T) + torch.matmul(text_embed_cap, text_embed_cat.T)) / 2)
                similarity_matrix_text = similarity_matrix_text.requires_grad_(False)
            #elif self.similarity_method == '':

            return similarity_matrix, similarity_matrix_text

        else:
            if self.text_model == 'bert-base-uncased':
                text_embed = self.text_encoder(input_ids=ids, attention_mask=masks).pooler_output

            elif self.text_model == 'roberta-base':
                text_embed = self.text_encoder(input_ids=ids, attention_mask=masks)[:, 0, :]
                
            text_embed = self.text_linear(text_embed)
            text_embed = text_embed / torch.norm(text_embed, dim=1, keepdim=True)

            # Compare similarity
            if self.similarity_method == 'cos_similarity':
                similarity_matrix = (torch.matmul(img_embed, text_embed.T) + torch.matmul(text_embed, img_embed.T)) / 2
            #elif self.similarity_method == '':

            return similarity_matrix

def nt_xent_loss(similarity_matrix, labels, temperature=0.1):
    # Calculate the loss for image-text pairs
    image_text_loss = F.cross_entropy(similarity_matrix / temperature, labels)

    # Calculate the loss for text-image pairs
    text_image_loss = F.cross_entropy(similarity_matrix.t() / temperature, labels)

    # Combine the losses and return the result
    return (image_text_loss + text_image_loss) / 2
import torch
import torch.nn as nn
import torchvision

from transformers import AutoModel



class Project_Embeddings_one_layer(nn.Module):
    def __init__(self, projection_dim, dropout_rate):
        super().__init__()
        self.gelu_linear_dropout = nn.Sequential(nn.GELU(),
                                                 nn.Linear(projection_dim, projection_dim),
                                                 nn.Dropout(dropout_rate))
        self.layer_norm = nn.LayerNorm(projection_dim)
        
    def forward(self, input_x):
        x = self.gelu_linear_dropout(input_x)
        #residual connection
        x = x + input_x
        x = self.layer_norm(x)
        return x



class Project_Embeddings(nn.Module):
    def __init__(self, num_projection_layers, hidden_dim, projection_dim, dropout_rate):
        super().__init__()
        self.first_layer = nn.Linear(hidden_dim, projection_dim)
        self.embedding_layers = nn.ModuleList([Project_Embeddings_one_layer(projection_dim, dropout_rate) for i in range(num_projection_layers)])
        
    def forward(self, embeddings):
        output = self.first_layer(embeddings)
        for layer in self.embedding_layers:
            output = layer(output)
        return output  




class Text_Encoder_BERT(nn.Module):
    def __init__(self, model_name, is_trainable, num_projection_layers, text_hidden_dim, projection_dim, dropout_rate):
        super().__init__()
        self.model_name = model_name
        
        self.text_encoder = AutoModel.from_pretrained(model_name) 
        
        for param in self.text_encoder.parameters():
            param.requires_grad = is_trainable
        
        self.projection = Project_Embeddings(num_projection_layers, text_hidden_dim, projection_dim, dropout_rate)
        
    def forward(self, input_ids, token_type_ids, attention_mask):

        text_inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask':attention_mask}

        bert_embedding = self.text_encoder(**text_inputs)
        
        cls_output = bert_embedding.pooler_output
        
        output = self.projection(cls_output)
        
        return output #torch.Size([batch_size, 256])




class Image_Encoder_ResNet(nn.Module):
    def __init__(self, is_trainable, num_projection_layers, img_hidden_dim, projection_dim, dropout_rate):
        super().__init__()
        
        resnet_image1 = torchvision.models.resnet50(pretrained=True)
        self.resnet_image1 = nn.Sequential(*list(resnet_image1.children())[:-1]) # remove linear classifier
        
        resnet_image2 = torchvision.models.resnet50(pretrained=True)
        self.resnet_image2 = nn.Sequential(*list(resnet_image2.children())[:-1])
        
        for param1, param2 in zip(self.resnet_image1.parameters(), self.resnet_image2.parameters()):
            param1.requires_grad = is_trainable
            param2.requires_grad = is_trainable
        
        self.projection = Project_Embeddings(num_projection_layers, img_hidden_dim, projection_dim, dropout_rate)
        
    def forward(self, image1, image2):
        # image_inputs = {'image1': tensor(B,3,128,128), 'image2': tensor(B,3,128,128)}
        
        output1 = self.resnet_image1(image1) # (B,C,1,1)
        output2 = self.resnet_image2(image2) # (B,C,1,1)
        
        concat = torch.cat((output1, output2), dim=1) # (B,2C,1,1)
                        
        batch_size = concat.size(0)
        output = concat.view((batch_size, 4096)) # (B,2C)
        
        output = self.projection(output)
        return output



class MultiModal_Classification(nn.Module):
    def __init__(self, text_model_name, is_text_trainable, is_img_trainable, num_projection_layers, text_hidden_dim, img_hidden_dim, projection_dim, dropout_rate, num_class):
        super().__init__()
        self.text_encoder = Text_Encoder_BERT(text_model_name, is_text_trainable, num_projection_layers, text_hidden_dim, projection_dim, dropout_rate)
        self.image_encoder = Image_Encoder_ResNet(is_img_trainable, num_projection_layers, img_hidden_dim, projection_dim, dropout_rate)
        
        self.classifier = nn.Linear(2*projection_dim, num_class)
        
    def forward(self, input_ids, token_type_ids, attention_mask, image1, image2):
        
        # Text encoder: BERT
        text_projection = self.text_encoder(input_ids, token_type_ids, attention_mask) # (B,256)
        
        # Image_encoder: ResNet
        image_projection = self.image_encoder(image1, image2) # (B,256)
        
        # Concat text and image projections
        text_image = torch.cat((text_projection, image_projection), dim=1) # (B,512)
        
        # Classifier
        output = self.classifier(text_image)
        
        return output   


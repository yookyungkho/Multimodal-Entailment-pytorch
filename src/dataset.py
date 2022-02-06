from PIL import Image

import torch
import torchvision

from transformers import AutoTokenizer



class MultiModal_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, img_size=128):
        
        # text
        self.text1 = df["text_1"].tolist()
        self.text2 = df["text_2"].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # image
        self.image1 = df["image_1_path"].tolist()
        self.image2 = df["image_2_path"].tolist()
        self.img_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(img_size, img_size)),
                                                             torchvision.transforms.ToTensor()])
        #label
        self.label = df["label_idx"].tolist()
    
    def __len__(self):
        return len(self.text1)
    
    def __getitem__(self, idx):
        # text inputs
        text1 = self.text1[idx]
        text2 = self.text2[idx]
        text_inputs = self.tokenizer(text1, text2, padding="max_length", truncation=True, max_length=128, return_tensors="pt") #batch별 max_len 자동 커스텀화
        
        # image inputs
        ## *Numbers of channels are different depending on image modes
        ## https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
        image1 = Image.open(self.image1[idx]).convert("RGB")
        image1 = self.img_transform(image1)
        image2 = Image.open(self.image2[idx]).convert("RGB")
        image2 = self.img_transform(image2)
        image_inputs = {'image1': image1, 'image2': image2}
        
        # labels
        labels = self.label[idx]
        
        return text_inputs, image_inputs, labels


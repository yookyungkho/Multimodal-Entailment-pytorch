import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch



#--------------------------------------------------------------------------


class CheckPoint:
    
    def __init__(self, dir_checkpoint, file_name):
        self.best_loss = 9999
        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)
        self.dir_checkpoint = dir_checkpoint
        self.file_name = file_name
    
    def step(self, model, log, epoch):
        if log['Valid Loss'] < self.best_loss:
            print(f"Model Save : Best Loss {self.best_loss:.4f} -> {log['Valid Loss']:.4f}")
            
            state = {'model_state_dict': model.state_dict(),
                     'loss': log['Valid Loss'],
                     'accuracy': log['Valid Acc'],
                     'f1_macro': log['Valid F1_macro'],
                     'f1_weighted': log['Valid F1_weighted'],
                     'epoch': epoch}

            torch.save(state, f'{self.dir_checkpoint}/{self.file_name}_multi_modal.pth')
        
            self.best_loss = log['Valid Loss']


#--------------------------------------------------------------------------


def load_data(dir_data, is_train):

    if is_train == True:
        train_df = pd.read_pickle(f"{dir_data}/train.pkl")
        valid_df = pd.read_pickle(f"{dir_data}/valid.pkl")
        return train_df, valid_df
    
    else:
        test_df = pd.read_pickle(f"{dir_data}/test.pkl")
        return test_df


#--------------------------------------------------------------------------


def visualize(idx):
    current_row = df.iloc[idx]
    image_1 = plt.imread(current_row["image_1_path"])
    image_2 = plt.imread(current_row["image_2_path"])
    text_1 = current_row["text_1"]
    text_2 = current_row["text_2"]
    label = current_row["label"]

    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.axis("off")
    plt.title("Image One")
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_2)
    plt.axis("off")
    plt.title("Image Two")
    plt.show()
    
    print("========="*5)
    print(f"Text1: {text_1}")
    print("========="*5)
    print(f"Text2: {text_2}")
    print("========="*5)
    print(f"Label: {label}")
    print("========="*5)


#--------------------------------------------------------------------------
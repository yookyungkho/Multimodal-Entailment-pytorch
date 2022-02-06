import os
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

import torch

from dataset import MultiModal_Dataset
from model import MultiModal_Classification
from trainer import Trainer
from utils import *


def train(cfg) -> None:
    # 1. Load Data
    train_df, val_df = load_data(cfg.dir_data, is_train=True)
    print(f">>>>>>> Complete: loading dataset!")
    print(f">>>>>>> Total training examples: {len(train_df)} \n>>>>>>> Total validation examples: {len(val_df)}")

    # Set enviormnet variable to prevent huggingface warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Fix random seed
    torch.manual_seed(cfg.seed_num)

    warnings.filterwarnings(action='ignore') 

    # 2. Create Dataset objects for train/validation sets.
    train_dataset = MultiModal_Dataset(df=train_df, img_size=cfg.img_size)
    valid_dataset = MultiModal_Dataset(df=val_df, img_size=cfg.img_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = cfg.batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = cfg.batch_size)

    print(f">>>>>>> Complete: create train, valid dataloader!")


    # 3. Load Multimodal Model, optimizer, criterion, Trainer
    model = MultiModal_Classification(
        cfg.text_model_name, cfg.is_text_trainable, cfg.is_img_trainable,
        cfg.num_projection_layers, cfg.text_hidden_dim, cfg.img_hidden_dim,
        cfg.project_dim, cfg.dropout_rate, cfg.num_class
        ).to(cfg.device)

    ## CE Loss
    criterion = torch.nn.CrossEntropyLoss()
    ## Optimizer : Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    trainer = Trainer(cfg, model, criterion, optimizer, train_dataloader, valid_dataloader)
    
    print(f">>>>>>> Complete: load model, optimizer, loss function, and Trainer!")

    # 4. Train Model
    print(">>>>>>> Start Training~!")
    print(">>>>>>> Remaining time will be shown in the tqdm bar below in few minutes.")
    
    trainer.train()




if __name__ == '__main__':
    print(f">>>>>>> Welcome to Multimodal Entailment! Let's start training process:)")

    parser = argparse.ArgumentParser(description='train_argparse')

    parser.add_argument("--dir_data", default="./data", type=str)
    parser.add_argument("--dir_checkpoint", default="./best_model", type=str)
    parser.add_argument("--file_name", default="exp1", type=str)

    parser.add_argument("--wandb_project_name", default="Multimodal Entailment", type=str)
    parser.add_argument("--wandb_entity_name", default="yookyungkho", type=str)

    parser.add_argument("--img_size", default=128, type=int)
    parser.add_argument("--text_model_name", default="bert-base-uncased", type=str)
    parser.add_argument("--is_text_trainable", default=False, type=bool)
    parser.add_argument("--is_img_trainable", default=False, type=bool)
    parser.add_argument("--project_dim", default=256, type=int)
    parser.add_argument("--num_projection_layers", default=2, type=int)
    parser.add_argument("--text_hidden_dim", default=768, type=int)
    parser.add_argument("--img_hidden_dim", default=4096, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--num_class", default=3, type=int)

    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed_num", default=0, type=int)
    
    parser.add_argument("--device", default='cuda', type=str)

    train_config = parser.parse_args()

    device = torch.device(train_config.device if torch.cuda.is_available() else 'cpu')
    print(f">>>>>>> Check Device: {device}")
    train_config.device = device


    train(train_config)
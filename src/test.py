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



def test(cfg) -> None:
    # 1. Load test data
    test_df = load_data(cfg.dir_data, is_train=False)
    print(f">>>>>>> Complete: load test data!")

    ## Set enviormnet variable to prevent huggingface warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ## Fix random seed
    torch.manual_seed(cfg.seed_num)

    warnings.filterwarnings(action='ignore') 

    # 2. Create Dataset objects for test sets.
    test_dataset = MultiModal_Dataset(df=test_df, img_size=cfg.img_size)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = cfg.batch_size)
    print(f">>>>>>> Complete: create test dataloader!")


    # 3. Load the best version of HAN model (lowest valid loss)
    weights = torch.load(f'{cfg.dir_checkpoint}/{cfg.file_name}_multi_modal.pth')
    
    model = MultiModal_Classification(
        cfg.text_model_name, cfg.is_text_trainable, cfg.is_img_trainable,
        cfg.num_projection_layers, cfg.text_hidden_dim, cfg.img_hidden_dim,
        cfg.project_dim, cfg.dropout_rate, cfg.num_class
        ).to(cfg.device)
    
    model.load_state_dict(weights['model_state_dict'])

    print('Load trained model')

    ## CE Loss
    criterion = torch.nn.CrossEntropyLoss()

    # 4. Load tester and get results
    tester = Trainer(cfg, model, criterion, is_train=False)

    test_log, pred_labels, real_labels = tester.evaluate_epoch(test_dataloader, is_valid=False)

    print(f">>>>>>> [Test] Loss: {test_log['Valid Loss']}, Accuracy: {test_log['Valid Acc']}")


    # 5. Save classification report and confusion matrix to txt file
    
    ## Defien label map
    label_map = {"Contradictory": 0, "Implies": 1, "NoEntailment": 2}
    class_names = list(label_map.keys())

    ## classification report
    report = classification_report(real_labels, pred_labels, target_names=class_names)
    ## confusion matrix
    cm = confusion_matrix(real_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    strings = f"Title: {cfg.file_name}\n\nClassification Report\n\n{report}\n\nConfusion Matrix\n\n{np.array2string(cm)}\n"
    
    with open(f"{cfg.dir_result}/{cfg.file_name}_test_cls_report_conf_matrix.txt", "w") as text_file:
        print(strings, file=text_file)

    print(f">>>>>>> Classification report:\n{report}")
    print(f">>>>>>> Confusion matrix:\n{cm_df}")
    print(f">>>>>>> Test result(classification report, confusion matrix) is saved as a txt file in the 'result' directory.")



if __name__ == '__main__':
    print(f">>>>>>> Let's start test process:)")

    parser = argparse.ArgumentParser(description='test_argparse')

    parser.add_argument("--dir_data", default="./data", type=str)
    parser.add_argument("--dir_checkpoint", default="./best_model", type=str)
    parser.add_argument("--dir_result", default="./result", type=str)
    parser.add_argument("--file_name", default="exp1", type=str)

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

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed_num", default=0, type=int)
    
    parser.add_argument("--device", default='cuda', type=str)

    test_config = parser.parse_args()

    device = torch.device(test_config.device if torch.cuda.is_available() else 'cpu')
    print(f">>>>>>> Check Device: {device}")
    test_config.device = device

    if not os.path.exists(test_config.dir_checkpoint):
        print(">>>>>>> (Warning) We didn't find out any trained model. Please run 'bash train.sh' first.")

    if not os.path.exists(test_config.dir_result):
        os.mkdir(test_config.dir_result)

    test(test_config)



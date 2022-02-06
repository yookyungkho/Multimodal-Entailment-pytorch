import wandb
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch

from utils import CheckPoint



class Trainer:
    def __init__(self, config, model, criterion, optimizer=None, train_dataloader=None, valid_dataloader=None, is_train=True):
        
        self.config = config
        self.model = model
        self.criterion = criterion
        
        if is_train==True:
            self.optimizer = optimizer
            self.train_dataloader = train_dataloader
            self.valid_dataloader = valid_dataloader

        self.device = self.config.device



    def train(self):
        # Wandb Settings for logging
        wandb.init(project=self.config.wandb_project_name, name=self.config.file_name, entity=self.config.wandb_entity_name, config=self.config)
        #wandb.config.update({"epochs": self.config.num_epochs,
        #                    "batch_size": self.config.batch_size,
        #                    "learning_rate" : self.config.learning_rate})
        wandb.watch(self.model, log="all")

        checkpoint = CheckPoint(self.config.dir_checkpoint, self.config.file_name)

        for epoch in tqdm(range(1, self.config.num_epochs+1)):
            
            # 1. Train
            train_log = self.train_epoch()
            print(f">>>>>>> [Epoch {epoch}]")
            print(f">>>>>>> [Train] Loss: {train_log['Train Loss']}, Accuracy: {train_log['Train Acc']}")

            # 2. Evaluate
            valid_log = self.evaluate_epoch(self.valid_dataloader, is_valid=True)
            print(f">>>>>>> [Valid] Loss: {valid_log['Valid Loss']}, Accuracy: {valid_log['Valid Acc']}, F1 macro: {valid_log['Valid F1_macro']}, F1 weighted: {valid_log['Valid F1_weighted']}")

            # 3. Save model if best loss is updated
            checkpoint.step(self.model, valid_log, epoch)

            # 4. Add log in wandb page
            wandb.log(train_log)
            wandb.log(valid_log)



    def train_epoch(self):
        self.model.train()

        # 0. Set initial loss and acc for each epoch
        losses = 0
        acc = 0
        total = 0

        for batch_idx, (text_inputs, image_inputs, labels) in enumerate(self.train_dataloader):
            # 1. Load batch and set device(cuda)
            ## text inputs
            input_ids = text_inputs['input_ids'].squeeze(1).to(self.config.device) # (B,1,128) -> (B,128)
            token_type_ids = text_inputs['token_type_ids'].squeeze(1).to(self.config.device)
            attention_mask = text_inputs['attention_mask'].squeeze(1).to(self.config.device)
            ## image inputs
            image1 = image_inputs['image1'].to(self.config.device)
            image2 = image_inputs['image2'].to(self.config.device)
            ## labels
            labels = labels.to(self.config.device)
            
            # 2. Get output from model
            output = self.model(input_ids,token_type_ids, attention_mask, image1, image2).to(self.config.device)
            
            # 3. Compute loss and update parameters
            self.optimizer.zero_grad()
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            losses += loss.item()

            # 4. Compute accuracy
            predicted = torch.max(output, 1)[1]
            total += labels.size(0)
            acc += (predicted == labels).sum().item()
            
                
        train_loss = losses / len(self.train_dataloader)
        train_acc = acc/total

        train_log = {'Train Loss': train_loss, 'Train Acc': train_acc}
            
        return train_log



    def evaluate_epoch(self, dataloader, is_valid=True):
        self.model.eval()
        # Set initial loss and acc
        losses = 0
        acc = 0
        total = 0
        f1_macro = 0
        f1_weighted = 0
        pred_labels, real_labels = [],[]
        
        with torch.no_grad():

            for batch_idx, (text_inputs, image_inputs, labels) in enumerate(dataloader):
                # 1. Load batch and set device(cuda)
                ## text inputs
                input_ids = text_inputs['input_ids'].squeeze(1).to(self.config.device) # (B,1,128) -> (B,128)
                token_type_ids = text_inputs['token_type_ids'].squeeze(1).to(self.config.device)
                attention_mask = text_inputs['attention_mask'].squeeze(1).to(self.config.device)
                ## image inputs
                image1 = image_inputs['image1'].to(self.config.device)
                image2 = image_inputs['image2'].to(self.config.device)
                ## labels
                labels = labels.to(self.config.device)
                
                # 2. Get output from model
                output = self.model(input_ids,token_type_ids, attention_mask, image1, image2).to(self.config.device)
                
                # 3. Compute accuracy and loss
                ## accuracy
                predicted = torch.max(output, 1)[1] # predicted label
                total += labels.size(0)
                acc += (predicted == labels).sum().item()
                ## loss
                loss = self.criterion(output, labels)
                losses += loss.item()

                # 4. Create numpy array of predictions and real labels to compute f1 scores
                pred = predicted.cpu().detach().numpy()
                real = labels.cpu().detach().numpy()

                # (only for test) Save predictions and real labels of each batch in the pred, real list
                if is_valid == False:
                    pred_labels.extend(pred)
                    real_labels.extend(real)

                # (only for validation) Compute f1 scores amd update recored scores
                if is_valid == True:
                    f1_macro_batch = f1_score(pred, real, average='macro')
                    f1_weighted_batch = f1_score(pred, real, average='weighted')
                    f1_macro += f1_macro_batch
                    f1_weighted += f1_weighted_batch

        valid_loss = losses / len(dataloader)
        valid_acc = acc/total

        if is_valid == True:
            valid_f1_macro = f1_macro / len(dataloader)
            valid_f1_weighted = f1_weighted / len(dataloader)

            valid_log = {'Valid Loss': valid_loss, 'Valid Acc': valid_acc, 'Valid F1_macro': valid_f1_macro, 'Valid F1_weighted': valid_f1_weighted}
            return valid_log
        else:
            test_log = {'Valid Loss': valid_loss, 'Valid Acc': valid_acc}

            return test_log, pred_labels, real_labels
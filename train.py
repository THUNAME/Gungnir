import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import torch.nn.functional as F
from Tokendict import Tokendict
from Config import config
from model_v2 import TransformerModel, load_model
from load_data import  tokenize_trainingdata, tokenize_test_data
from Tools import create_autoregressive_mask,left_pad_sequences
from Dataset import MyDataset
from torch.nn.utils.rnn import pad_sequence

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss




def train():
    # Set random seed to ensure reproducibility of results
    torch.manual_seed(0)
    np.random.seed(0)
    # Set device to GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    tokendict=Tokendict(config=config)
    csv_list=tokendict.load_csv_data(config=config)
    
    tokendict.save_vocab(config=config)
    
    
    tokenized_Dataset = []
    for csv_data in csv_list:
        # 0:asnum, 1:org_name, 2:category, 3:sub_category, 4:routing_prefix, 5:prefix, 6:active_type
        csv_data=[csv_data[6],csv_data[2],csv_data[3],csv_data[1],csv_data[0],csv_data[4],csv_data[5]]
        # 6:active_type, 2:category, 3:sub_category, 1:org_name, 0:asnum, 4:routing_prefix, 5:prefix
        
        tokenized_data = tokenize_trainingdata(csv_data, tokendict.vocab)
        tokenized_data = tokenize_trainingdata(csv_data, tokendict.vocab)
        tokenized_Dataset.append(tokenized_data)
    
    
    # # Calculate the length of each sequence
    # input_seqlens = [len(tokenized_Dataset[i]) for i in range(len(tokenized_Dataset))]
    # # Left pad the sequences to make all sequences have the same length
    # tokenized_Dataset = left_pad_sequences(tokenized_Dataset, tokendict.vocab)
    
    
    # Define dataset and data loader
    dataset = MyDataset(tokenized_Dataset,tokendict.vocab)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True,collate_fn=collate_fn)


    model = load_model(config=config,tokendict=tokendict)
    criterion = nn.CrossEntropyLoss(ignore_index=tokendict.vocab["PAD_TOKEN"]).to(device)  # Use ignore_index to ignore padding token
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1
    for epoch in range(num_epochs):
        running_loss = 0.0
        for input_sequences_left, input_sequences_left_mask, target_sequences_left, target_sequences_left_mask in dataloader:
            
            batch_size, seq_len = input_sequences_left.size()
            mask = create_autoregressive_mask(seq_len)
            
            optimizer.zero_grad()
            outputs = model(input_sequences_left, mask , src_key_padding_mask = input_sequences_left_mask)
            # Since the output is [batch_size, seq_len, output_size]
            # We need to reshape outputs and batch_labels to fit CrossEntropyLoss
            outputs = outputs.view(-1, model.output_size)  # [batch_size * seq_len, output_size]
            
            target_sequences_left = target_sequences_left.reshape(-1)  # [batch_size * seq_len]
            
            loss = criterion(outputs, target_sequences_left)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        logging.info(f'Epoch {epoch + 1} Loss: {running_loss / len(dataloader)}')
    
        # Assuming the model training is complete, model is your TransformerModel instance
        torch.save(model.state_dict(), config["model_path"]+".pth")
    logging.info('Model saved!')
    return model



def collate_fn(batch):
    # Add START_TOKEN at the beginning of each sample
    # vocab["START_TOKEN"]=2
    # vocab["PAD_TOKEN"]=5
    
    device = batch[0].device  # Get the device of the first sample
    sequences = [torch.cat([
        torch.tensor([2], device=device),  # Add start token and ensure it's on the correct device
        sample.to(device)  # Move the sample to the correct device
    ]) for sample in batch]
    
    
    # Left padding
    reversed_seqs = [torch.flip(seq, [0]) for seq in sequences]  # First reversal
    padded_reversed = pad_sequence(reversed_seqs, 
                                 batch_first=True, 
                                 padding_value=5)  # Right padding the reversed sequences
    padded_sequences_left = torch.flip(padded_reversed, [1])  # Second reversal
    
    input_sequences_left = padded_sequences_left[:, :-1]  # Extract input sequences
    target_sequences_left = padded_sequences_left[:, 1:]  # Extract target sequences
    
    
    input_sequences_left_mask = (input_sequences_left != 5).float()
    target_sequences_left_mask = (target_sequences_left != 5).float()
    
    
    
    return input_sequences_left, input_sequences_left_mask, target_sequences_left, target_sequences_left_mask



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()
    model = train()
    end_time = time.time()
    logging.info("Training time: %f seconds", end_time - start_time)
    

    
    
    
    
    

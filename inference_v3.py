import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math
import torch.nn.functional as F
from load_data import tokenize_test_data, untokenize_data
import csv
from Config import config
from Tokendict import Tokendict
from model_v2 import load_model
from Tools import create_autoregressive_mask,add_ipv6_colon,left_pad_sequences,left_pad_sequences_inference
from Environment import Environment
from Dataset import MyDataset, TestDataset
import logging





def nucleus_sampling(probs, top_p):

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Determine the tokens to be removed
    mask = cumulative_probs >= top_p
    has_met = mask.any(dim=-1, keepdim=True)
    # Find the first index that satisfies the condition
    i = torch.where(
        has_met,
        mask.int().argmax(dim=-1, keepdim=True),
        torch.tensor(sorted_probs.size(-1), dtype=torch.long, device=probs.device)
    )
    # Generate a mask for the indices to be removed: positions where index > i
    sorted_indices_to_remove = torch.arange(sorted_probs.size(-1), device=probs.device) > i
    
    # Set the probabilities of the tokens to be removed to zero
    batch_size, vocab_size = probs.shape
    rows = torch.arange(batch_size, device=probs.device).view(-1, 1).expand(-1, vocab_size)
    cols = sorted_indices
    probs[rows[sorted_indices_to_remove], cols[sorted_indices_to_remove]] = 0
    
    # Renormalize the probability distribution
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    return probs

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
                                 padding_value=5)  # Right pad the reversed sequences
    padded_sequences_left = torch.flip(padded_reversed, [1])  # Second reversal
    
    mask = (padded_sequences_left != 5).float()
    return padded_sequences_left, mask



def predict_FRP(  
        config,
        vocab,
        model,
        # Inference example
        # 5:active_type, 2:category, 3:sub_category, 1:org_name, 0:asnum, 4:routing_prefix
        tokenized_Dataset,
        max_length=20,
        temperature=1.0,# The higher the temperature, the more diverse the generated tokens
        top_p=0.5,# The larger the top_p, the more diverse the generated tokens
    ):

    
    # Calculate the length of each sequence
    input_seqlens = [len(tokenized_Dataset[i]) for i in range(len(tokenized_Dataset))]
    
    # Define the dataset and data loader
    dataset = TestDataset(tokenized_Dataset, vocab)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False,collate_fn=collate_fn)  # No need to shuffle

    generated_sequences = []
    genarate_FRPs=set()
    
    for batch_seq,mask in dataloader:
        src_key_padding_mask = (mask == 0)  # True indicates the position that needs to be masked
        batch_size, seq_len = batch_seq.size()
        
        with torch.no_grad():

            for _ in range(max_length - 1):
                
                outputs = model(batch_seq,src_key_padding_mask=src_key_padding_mask)
                outputs = outputs[:, -1, :]  # Get the output of the last time step
                probs = F.softmax(outputs / temperature, dim=-1)  # Calculate softmax probability and apply temperature parameter
                
                probs = nucleus_sampling(probs, top_p)  
                next_token = torch.multinomial(probs, 1) # Randomly select a token based on probability
                
                batch_seq = torch.cat((batch_seq, next_token), dim=1)    
                src_key_padding_mask = torch.cat([src_key_padding_mask, torch.tensor([[False]]*len(src_key_padding_mask), device='cuda:0')], dim=1) 
            

            # Remove padding tokens
            # Create a mask where the position of padding tokens is False, and other positions are True
            mask = batch_seq != vocab["PAD_TOKEN"]
            
            # Extract non-padded elements for each sample
            non_padded_list = [seq[mask[i]] for i, seq in enumerate(batch_seq)]
            
            
            for generated_sequence in non_padded_list:
                # Move the CUDA tensor to CPU
                
                # 5:active_type, 2:category, 3:sub_category, 1:org_name, 0:asnum, 4:routing_prefix
                tensor_data_cpu = generated_sequence[6:].cpu()
                # Convert to list
                int_list = tensor_data_cpu.tolist()
                # Convert integer list to string
                str_representation = ''.join(map(str, int_list))
                genarate_FRPs.add(str_representation)


                
        
    return genarate_FRPs
            


if __name__ == '__main__':
    
    logger = logging.getLogger(__name__)
    
    # GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokendict = Tokendict(config=config)
    csv_list = tokendict.load_csv_data(config=config)

    model = load_model(config, tokendict)
    model.load_state_dict(torch.load(config["model_path"]+".pth"))  
    
    
    tokenized_Dataset=[]
    
    # a_1 = ["38794","Renaissance Phuket Resort & Spa","Computer and Information Technology","Internet Service Provider (ISP)","00100100000000001100110100000000","0010010000000000110011010000000000010000010001010000000000000000","TCP_80"]
    # b_1 = tokenize_test_data(a_1, tokendict.vocab)
    # tokenized_Dataset.extend([b_1]*100)
    
    # 24356,CERNET2 IX at Chongqing University,Education and Research,"Colleges, Universities, and Professional Schools",001000000000000100000010010100000010010000011010,TCP_443
    a_2 = ["58810","iZus Co., Ltd","Computer and Information Technology","Hosting and Cloud Provider","001001000000000111001111100000000110000001000100","TCP_443"]
    b_2 = tokenize_test_data(a_2, tokendict.vocab)
    
    time1=time.time()
    tokenized_Dataset.extend([b_2]*200)

    
    environment = Environment(config)
    
    frps = predict_FRP(config, tokendict.vocab, model, tokenized_Dataset)

    time2=time.time()
    logging.info("time:",time2-time1)
    
    for frp in frps:
        ipv6_address, bin_length = untokenize_data(frp, tokendict.vocab, logger ,environment)
        if ipv6_address != None:
            ipv6_address = add_ipv6_colon(ipv6_address)
            logging.info(ipv6_address + "/" + str(bin_length))

    

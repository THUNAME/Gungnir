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
    """
    核采样（Nucleus Sampling）
    
    参数:
    probs (torch.Tensor): 概率分布张量，形状为(batch_size, vocab_size)
    top_p (float): 累积概率阈值
    
    返回:
    torch.Tensor: 经过核采样处理后的概率分布张量
    """
    # 对概率进行降序排序
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 确定需要移除的token
    mask = cumulative_probs >= top_p
    has_met = mask.any(dim=-1, keepdim=True)
    # 找到第一个满足条件的索引
    i = torch.where(
        has_met,
        mask.int().argmax(dim=-1, keepdim=True),
        torch.tensor(sorted_probs.size(-1), dtype=torch.long, device=probs.device)
    )
    # 生成需要移除的mask：索引>i的位置
    sorted_indices_to_remove = torch.arange(sorted_probs.size(-1), device=probs.device) > i
    
    # 将需要移除的token对应的概率置零
    batch_size, vocab_size = probs.shape
    rows = torch.arange(batch_size, device=probs.device).view(-1, 1).expand(-1, vocab_size)
    cols = sorted_indices
    probs[rows[sorted_indices_to_remove], cols[sorted_indices_to_remove]] = 0
    
    # 重新归一化概率分布
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    return probs

def collate_fn(batch):
    # 在每个样本开头添加START_TOKEN
    # vocab["START_TOKEN"]=2
    # vocab["PAD_TOKEN"]=5
    
    device = batch[0].device  # 获取第一个样本的设备
    sequences = [torch.cat([
        torch.tensor([2], device=device),  # 添加起始标记，并确保它在正确的设备上
        sample.to(device)  # 将样本移动到正确的设备上
    ]) for sample in batch]
    
    
    # 左填充
    reversed_seqs = [torch.flip(seq, [0]) for seq in sequences]  # 第一次反转
    padded_reversed = pad_sequence(reversed_seqs, 
                                 batch_first=True, 
                                 padding_value=5)  # 右侧填充反转后的序列
    padded_sequences_left = torch.flip(padded_reversed, [1])  # 第二次反转
    
    mask = (padded_sequences_left != 5).float()
    return padded_sequences_left, mask



def predict_FRP(  
        config,
        vocab,
        model,
        # 推理示例
        # 5:active_type, 2:category, 3:sub_category, 1:org_name, 0:asnum, 4:routing_prefix
        tokenized_Dataset,
        max_length=20,
        temperature=1.0,# temperature越高,生成的token越多样化
        top_p=0.5,# top_p越大,生成的token越多样化
    ):

    
    # 计算每个序列的长度
    input_seqlens = [len(tokenized_Dataset[i]) for i in range(len(tokenized_Dataset))]
    
    # 定义数据集和数据加载器
    dataset = TestDataset(tokenized_Dataset, vocab)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False,collate_fn=collate_fn)  # 不需要shuffle

    generated_sequences = []
    genarate_FRPs=set()
    
    for batch_seq,mask in dataloader:
        src_key_padding_mask = (mask == 0)  # True表示需要mask的位置
        batch_size, seq_len = batch_seq.size()
        
        with torch.no_grad():

            for _ in range(max_length - 1):
                
                outputs = model(batch_seq,src_key_padding_mask=src_key_padding_mask)
                outputs = outputs[:, -1, :]  # 获取最后一个时间步的输出
                probs = F.softmax(outputs / temperature, dim=-1)  # 计算softmax概率并应用温度参数
                
                probs = nucleus_sampling(probs, top_p)  
                next_token = torch.multinomial(probs, 1) # 根据概率随机选择一个token
                
                batch_seq = torch.cat((batch_seq, next_token), dim=1)    
                src_key_padding_mask = torch.cat([src_key_padding_mask, torch.tensor([[False]]*len(src_key_padding_mask), device='cuda:0')], dim=1) 
            

            # 移除填充token
            # 创建一个掩码，其中填充token的位置为False，其他位置为True
            mask = batch_seq != vocab["PAD_TOKEN"]
            
            # 逐样本提取非填充元素
            non_padded_list = [seq[mask[i]] for i, seq in enumerate(batch_seq)]
            
            
            for generated_sequence in non_padded_list:
                # 将CUDA张量移动到CPU
                
                # 5:active_type, 2:category, 3:sub_category, 1:org_name, 0:asnum, 4:routing_prefix
                tensor_data_cpu = generated_sequence[6:].cpu()
                # 转换为列表
                int_list = tensor_data_cpu.tolist()
                # 将整数列表转换为字符串
                str_representation = ''.join(map(str, int_list))
                genarate_FRPs.add(str_representation)


                
        
    return genarate_FRPs
            


if __name__ == '__main__':
    
    logger = logging.getLogger(__name__)
    
    # 设置设备为GPU或CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokendict = Tokendict(config=config)
    csv_list = tokendict.load_csv_data(config=config)

    model = load_model(config, tokendict)
    model.load_state_dict(torch.load(config["model_path"]+".pth"))  # 加载模型参数
    
    
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
        # 将生成的token序列转换为原始的标签
        ipv6_address, bin_length = untokenize_data(frp, tokendict.vocab, logger ,environment)
        if ipv6_address != None:
            ipv6_address = add_ipv6_colon(ipv6_address)
            logging.info(ipv6_address + "/" + str(bin_length))

    

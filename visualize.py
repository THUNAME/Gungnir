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
from model import TransformerModel, load_model
from Tools import create_autoregressive_mask
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



def visualize_embeddings(vocab, model, tokens, device):
    # 确保所有token都是字符串
    tokens = [str(token) for token in tokens]
    
    # 检查token是否在词汇表中
    valid_tokens = [token for token in tokens if token in vocab]
    if len(valid_tokens) != len(tokens):
        invalid_tokens = [token for token in tokens if token not in vocab]
        print(f"Warning: The following tokens are not in the vocabulary and will be ignored: {invalid_tokens}")

    # 获取token的embedding
    token_indices = torch.tensor([vocab[token] for token in valid_tokens], dtype=torch.long).to(device)
    embeddings = model.embedding(token_indices).detach().cpu().numpy()  # 添加detach()方法

    # 使用t-SNE进行降维
    n_samples = len(embeddings)
    if n_samples < 2:
        print("Not enough samples to perform t-SNE. Need at least 2 samples.")
        return

    # 设置perplexity为小于n_samples的值
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # 绘制t-SNE结果
    plt.figure(figsize=(20, 10))
    for i, token in enumerate(valid_tokens):
        plt.scatter(embeddings_tsne[i, 0], embeddings_tsne[i, 1], label=token)
        plt.annotate(token, (embeddings_tsne[i, 0], embeddings_tsne[i, 1]))
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Token Embeddings')
    # plt.legend()
    plt.savefig('tsne_visualization.png')



if __name__ == '__main__':
    # 设置设备为GPU或CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokendict = Tokendict(config=config)
    csv_list = tokendict.load_csv_data(config=config)

    model = load_model(config, tokendict)
    model.load_state_dict(torch.load(config["model_path"]))  # 加载模型参数
    model.to(device)  # 将模型移动到指定设备

    # # 示例tokens
    # tokens = [7545, "TPG Internet Pty Ltd", "Computer and Information Technology", "Internet Service Provider (ISP)", "0010000000000001010001000111100110100010"]

    tokens=tokendict.vocab.keys()
    # 可视化这些token的embedding
    tokens = list(tokens)
    new_tokens=[]
    for token in tokens:
        token=str(token)
        if token.isdigit():
            new_tokens.append(token)

    visualize_embeddings(tokendict.vocab, model, new_tokens, device)
    
    

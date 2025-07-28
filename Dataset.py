
import torch
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, sequences,vocab):
        # 初始化数据集，传入序列和词汇表
        self.sequences = sequences
        self.vocab = vocab
        # 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        # 返回数据集中序列的数量
        return len(self.sequences)

    def __getitem__(self, idx):
        # 根据索引获取对应的序列
        sequence = self.sequences[idx]
        # 在序列开头添加开始token，结尾添加结束token
        input_seq = list(map(int, sequence))
        
        input_seq = torch.tensor(input_seq, dtype=torch.long).to(self.device)  # 将输入序列移动到设备
        
        return input_seq


class TestDataset(Dataset):
    def __init__(self, sequences,vocab):
        # 初始化数据集，传入序列和词汇表
        self.sequences = sequences
        self.vocab = vocab
        # 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        # 返回数据集中序列的数量
        return len(self.sequences)

    def __getitem__(self, idx):
        # 根据索引获取对应的序列
        sequence = self.sequences[idx]
        # 在序列开头添加开始token，结尾添加结束token
        input_seq =  list(map(int, sequence))
        
        
        input_seq = torch.tensor(input_seq, dtype=torch.long).to(self.device)  # 将输入序列移动到设备
        return input_seq










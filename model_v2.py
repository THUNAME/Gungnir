from typing import Optional
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_heads, 
                 num_layers, 
                 output_size,
                 max_len=5000,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size).to(device)
        self.positional_encoding = PositionalEncoding(hidden_size, max_len=max_len).to(device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device

    def forward(self, x, mask: Optional[torch.Tensor] = None, src_key_padding_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        x = self.fc(x)
        
        return x

def load_model(config, tokendict):
    # 模型参数设置
    input_size = tokendict.size()  # 因为输入是0、1，加上开始、结束和填充token
    output_size = tokendict.size()  # 输出是0、1，加上开始、结束和填充token
    hidden_size = config["hidden_size"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    
    model = TransformerModel(input_size, hidden_size, num_heads, num_layers, output_size)

    return model

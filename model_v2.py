from typing import Optional
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Initialize positional encoding matrix with zeros
        pe = torch.zeros(max_len, d_model)
        # Create a tensor with position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Calculate the divisor term for positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # Apply sine to even indices in the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the positional encoding matrix
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add batch dimension and transpose for broadcasting
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register the positional encoding matrix as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input tensor
        x = x + self.pe[:x.size(0), :]
        return x

# TransformerModel: A Transformer model based on PyTorch
# This model includes an embedding layer, positional encoding, transformer encoder layers, and a fully connected layer.
class TransformerModel(nn.Module):
    def __init__(self, 
                 input_size,  # The size of the input vocabulary
                 hidden_size,  # The number of features in the hidden state
                 num_heads,  # The number of attention heads in the transformer encoder
                 num_layers,  # The number of transformer encoder layers
                 output_size,  # The size of the output vocabulary
                 max_len=5000,  # The maximum length of the input sequences
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):  # The device to run the model on
        
        super(TransformerModel, self).__init__()
        
        # Embedding layer: Converts input tokens to dense vectors of fixed size
        self.embedding = nn.Embedding(input_size, hidden_size).to(device)
        
        # Positional Encoding: Adds information about the relative or absolute position of tokens in the sequence
        self.positional_encoding = PositionalEncoding(hidden_size, max_len=max_len).to(device)
        
        # Transformer Encoder Layer: A single layer of the transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True).to(device)
        
        # Transformer Encoder: Stacks multiple transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)
        
        # Fully Connected Layer: Maps the hidden state to the output size
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        
        # Store model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device

    def forward(self, x, mask: Optional[torch.Tensor] = None, src_key_padding_mask=None):
        # Embedding: Convert input tokens to dense vectors
        x = self.embedding(x)
        
        # Positional Encoding: Add positional information to the embeddings
        x = self.positional_encoding(x)
        
        # Transformer Encoder: Process the embedded and positional encoded input
        x = self.transformer_encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        # Fully Connected Layer: Map the transformer encoder output to the output size
        x = self.fc(x)
        
        return x

def load_model(config, tokendict):
    # Model parameter settings
    input_size = tokendict.size()  # Because the input is 0 and 1, add start, end, and padding tokens
    output_size = tokendict.size()  # The output is 0 and 1, add start, end, and padding tokens
    hidden_size = config["hidden_size"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    
    model = TransformerModel(input_size, hidden_size, num_heads, num_layers, output_size)

    return model

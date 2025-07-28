
import torch
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, sequences, vocab):
        # Initialize the dataset with sequences and vocabulary
        self.sequences = sequences
        self.vocab = vocab
        # Check if GPU is available, use GPU if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        # Return the number of sequences in the dataset
        return len(self.sequences)

    def __getitem__(self, idx):
        # Get the sequence corresponding to the index
        sequence = self.sequences[idx]
        # Add start token at the beginning and end token at the end of the sequence
        input_seq = list(map(int, sequence))
        
        input_seq = torch.tensor(input_seq, dtype=torch.long).to(self.device)  # Move the input sequence to the device
        
        return input_seq


class TestDataset(Dataset):
    def __init__(self, sequences, vocab):
        # Initialize the dataset with sequences and vocabulary
        self.sequences = sequences
        self.vocab = vocab
        # Check if GPU is available, use GPU if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        # Return the number of sequences in the dataset
        return len(self.sequences)

    def __getitem__(self, idx):
        # Get the sequence corresponding to the index
        sequence = self.sequences[idx]
        # Add start token at the beginning and end token at the end of the sequence
        input_seq = list(map(int, sequence))
        
        
        input_seq = torch.tensor(input_seq, dtype=torch.long).to(self.device)  # Move the input sequence to the device
        return input_seq










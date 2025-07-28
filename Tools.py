import torch
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_autoregressive_mask(seq_len):
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1).to(device)
    return mask

def left_pad_sequences(sequences, vocab):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_length = max_length - len(seq)
        # padded_seq = [vocab["PAD_TOKEN"]] * pad_length + [vocab["START_TOKEN"]]+ seq + [vocab["END_TOKEN"]]
        padded_seq = [vocab["PAD_TOKEN"]] * pad_length + [vocab["START_TOKEN"]]+ seq
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences)


def left_pad_sequences_inference(sequences, vocab):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_length = max_length - len(seq)
        padded_seq = [vocab["PAD_TOKEN"]] * pad_length + [vocab["START_TOKEN"]]+ seq
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences)

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
                                 padding_value=5)  # Right padding for the reversed sequences
    padded_sequences_left = torch.flip(padded_reversed, [1])  # Second reversal
    
    mask = (padded_sequences_left != 5).float()
    return padded_sequences_left, mask

    

def add_new_data_to_df(existing_df, new_data):
    """
    Add new data to an existing DataFrame and return the updated DataFrame.
    
    Parameters:
    - existing_df: The existing DataFrame to which new data will be added.
    - new_data: A dictionary containing the new data to be added.
    
    Returns:
    - updated_df: The updated DataFrame.
    """
    # Convert new data to DataFrame
    new_df = pd.DataFrame(new_data)
    
    # Add new data to existing DataFrame
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    return updated_df


def add_ipv6_colon(ipv6):
    # add : to ipv6
    ipv6_list = []
    for i in range(0, len(ipv6), 4):
        ipv6_list.append(ipv6[i:i+4])
    ipv6 = ":".join(ipv6_list)
    return ipv6



if __name__ == '__main__':
    pass
    
    
    
    
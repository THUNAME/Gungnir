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

    

def add_new_data_to_df(existing_df, new_data):
    """
    Add new data to an existing DataFrame and return the updated DataFrame.
    
    Parameters:
    - existing_df: The existing DataFrame to which new data will be added.
    - new_data: A dictionary containing the new data to be added.
    
    Returns:
    - updated_df: The updated DataFrame.
    """
    # 将新数据转换为DataFrame
    new_df = pd.DataFrame(new_data)
    
    # 将新数据添加到现有DataFrame
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
    
    
    
    
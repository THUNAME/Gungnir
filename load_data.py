import csv
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from Tokendict import Tokendict
from Config import config


def tokenize_trainingdata(untokenization_data,vocab):
    sequence=[]
    
    # asnum,org_name,category,sub_category,routing_prefix,prefix
    for data in untokenization_data:
        if data in vocab:
            sequence.append(vocab[data])
    
    assert len(untokenization_data)==7
    
    prefix = untokenization_data[-1]
    for i in range(len(prefix)):
        sequence.append(int(prefix[i]))
        
    return sequence
    
def tokenize_test_data(untokenization_data,vocab):
    sequence=[]
    
    # 5:active_type, 2:category, 3:sub_category, 1:org_name, 0:asnum, 4:routing_prefix
    
    for idx in [5,2,3,1,0]:
        if untokenization_data[idx] in vocab:
            sequence.append(vocab[untokenization_data[idx]])
        else:
            sequence.append(vocab["UNSEED_TOKEN"])
            

    prefix = untokenization_data[4]
    
    for i in range(len(prefix)):
        sequence.append(int(prefix[i]))
            
    return sequence


def untokenize_data(tokenization_data, vocab, logger, environment):
    # Convert the tokenization data to a list and append the END_TOKEN from the vocabulary
    tokenization_data = list(tokenization_data)
    tokenization_data.append(str(vocab["END_TOKEN"]))
    
    try:
        # Extract the binary number from the tokenization data up to the first occurrence of "3"
        bin_num = tokenization_data[:tokenization_data.index("3")]
        # Get the length of the binary number, limited to 128 characters
        bin_length = min(len(bin_num), 128)
        # Convert the binary number to a string
        bin_num = "".join(str(x) for x in bin_num)
        
        # Check if the binary number is a subset of the environment or has been checked
        if environment.issubset(bin_num) or environment.is_checked(bin_num):
            return None, 0
        else:
            # Add the binary number to the known_Area of the environment
            environment.known_Area.add(bin_num)
            
        # Pad the binary number to 128 characters with "0"
        bin_num = bin_num[:128] + (128 - bin_length) * "0"
        
        # Convert the binary number to a decimal number
        decimal_num = int(bin_num, 2)
        # Convert the decimal number to a hexadecimal string
        hex_num = "{:X}".format(decimal_num)

        return hex_num, bin_length
    
    except Exception as e: 
        # Log any exceptions that occur
        logger.error(f"An error occurred: {e}", exc_info=True)
        return None, 0

if __name__ == '__main__':
    
    tokendict = Tokendict(config=config)
    csv_list = tokendict.load_csv_data(config=config)
    
    untokenization_data=['262191', 'COLUMBUS NETWORKS COLOMBIA', 'Computer and Information Technology','Internet Service Provider (ISP)', '001000000000000000001011011100000000000000100101']
    a=tokenize_test_data(untokenization_data,tokendict.vocab)
    print(a)
    
    
    
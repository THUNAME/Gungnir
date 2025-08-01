

import csv
from Config import config

class Tokendict:
    def __init__(self,config):
        self.vocab = {}
        self.config = config
        
        self.vocab[0]=0
        self.vocab[1]=1
        self.vocab["START_TOKEN"]=2 # Define start and end tokens
        self.vocab["END_TOKEN"]=3
        self.vocab["UNSEED_TOKEN"]=4
        self.vocab["PAD_TOKEN"]=5
        
    def size(self):
        return len(self.vocab)
        
        
    def load_csv_data(self,config):
        
        with open(config["dataset"]) as data:
            csv_lists = []
            reader = csv.reader(data)  # Use file object directly
            
            header = next(reader)  # Read header row
            for row in reader:
                csv_lists.append(row)  # Remove extra quotes and spaces around tokens

        # 0:as,1:org_name,2:category,3:sub_category,4:routing_prefix,5:prefix,6:active_type
        # Combine lists
        combined_list = []
        for csv_list in csv_lists:
            combined_list.extend([csv_list[i] for i in [0,1,2,3,6]])
            
        # combined_list = [x for sub_list in csv_list for x in sub_list[0,1,2,3,6]]
        
        # Next available token_id, initialized to 0
        next_available_token_id = config["start_token_id"]


        for token in combined_list:
            if token not in self.vocab:
                self.vocab[token] = next_available_token_id
                next_available_token_id += 1
        
        return csv_lists
    
    def save_vocab(self,config):
        # Optionally save the vocabulary (e.g., save as a CSV file for easy viewing and future use)
        with open(config["vocab_save_path"], 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for token, token_id in self.vocab.items():
                writer.writerow([token, token_id])
                


if __name__ == '__main__':
    tokendict=Tokendict(config=config)
    csv_list=tokendict.load_csv_data(config=config)
    tokendict.save_vocab(config=config)
    print(csv_list)
    
    
    
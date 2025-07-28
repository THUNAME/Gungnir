
import ipaddress
import pandas as pd
import torch
import csv
from model import TransformerModel, load_model
from Tokendict import Tokendict
from Config import config   
from inference import predict_FRP


def load_lookuptable(filename):
    df = pd.read_csv(filename)
    as_org_category_subcategory=df[["as","org_name","category","sub_category"]]
    return as_org_category_subcategory

def trans_into_bin(routingprefix):
    address,length = routingprefix.split("/")
    length=int(length)
    ipv6_object = ipaddress.IPv6Address(address)
    routing_binary_representation = format(int(ipv6_object), '0128b')
    return routing_binary_representation[:length]

def make_Prediction(config,tokendict,model):
    

    lookuptable=load_lookuptable(config["lookuptablefile"])
    
    file = open(config["routingprefix_file"])

    
    with open(config["Prediction_path"], 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for line in file:
            try:
                line=line.strip()
                routingprefix,as_num=line.split("	")
                found=lookuptable[lookuptable['as'] == int(as_num)]
                # 0:asnum, 1:org_name, 2:category, 3:sub_category, 4:routing_prefix, 5:prefix, 6:active_type
                if found.empty:
                    input_token=[as_num,"PAD_TOKEN","PAD_TOKEN","PAD_TOKEN"]
                    input_token.append(trans_into_bin(routingprefix))
                else:
                    input_token=found.iloc[0].tolist()
                    input_token.append(trans_into_bin(routingprefix))
                input_token.append("TCP_443")
                # 逐行写入列表中的数据
                writer.writerow(input_token)
            except Exception as e:
                print(line)
                print(e)
            

    csvfile.close()
    file.close()
        
        
        


if __name__ == '__main__':

    tokendict=Tokendict(config=config)
    csv_list=tokendict.load_csv_data(config=config)
    make_Prediction(config=config,tokendict=tokendict,model=None)





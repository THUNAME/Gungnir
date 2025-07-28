import ipaddress
import pandas as pd
import torch
import csv
from model_v2 import TransformerModel, load_model
from Tokendict import Tokendict
from Config import config
from inference_v3 import predict_FRP
import logging
import os
from Environment import Environment
from multiprocessing import set_start_method, Pool, cpu_count
from load_data import tokenize_test_data, untokenize_data
from Tools import add_ipv6_colon
import time

# 设置多进程的启动方法为'spawn'
set_start_method('spawn', force=True)

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_lookuptable(filename):
    try:
        df = pd.read_csv(filename)
        as_org_category_subcategory = df[["category","sub_category", "org_name", "as" ]]
        logger.info(f"Successfully loaded lookup table from {filename}")
        return as_org_category_subcategory
    except Exception as e:
        logger.error(f"Failed to load lookup table from {filename}: {e}")
        return None

def trans_into_bin(routingprefix):
    try:
        address, length = routingprefix.split("/")
        length = int(length)
        ipv6_object = ipaddress.IPv6Address(address)
        routing_binary_representation = format(int(ipv6_object), '0128b')
        logger.debug(f"Routing prefix {routingprefix} converted to binary: {routing_binary_representation[:length]}")
        return routing_binary_representation[:length]
    except Exception as e:
        logger.error(f"Failed to convert routing prefix {routingprefix} to binary: {e}")
        return None
    

def do_Strategy(config, tokendict, models):

    tokenized_Dataset = []
    
    try:
        with open(config["Prediction_path"], 'r') as file:
            logger.info(f"Opened {config['Prediction_path']} for reading")
            lines = file.readlines()
            environment = Environment(config)
            
            for line in lines:
                reader = csv.reader([line])
                input_token = next(reader)
                tokenized_Dataset.extend([tokenize_test_data(input_token, tokendict.vocab)]*config["num_genarate_FRP"])
            
            frps = predict_FRP(config, tokendict.vocab, models[0], tokenized_Dataset,top_p=0.6,max_length=20)
            file = open(config["genarate_FRP_path"], "a")
            for frp in frps:
                ipv6_address, bin_length = untokenize_data(frp, tokendict.vocab, logger ,environment)
                if ipv6_address != None:
                    ipv6_address = add_ipv6_colon(ipv6_address)
                    file.write(ipv6_address + "/" + str(bin_length) + "\n")
            file.close()
            
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

def load_models(config, tokendict):
    try:
        models = []
        for i in range(config["num_models"]):
            model = load_model(config, tokendict)
            model.load_state_dict(torch.load(config["model_path"]+".pth"))  # 加载模型参数
            model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # 设置设备
            logger.info(f"Model {i} loaded from {config['model_path']}_model_{i}.pt")
            models.append(model)
        return models
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return None

def main():
    
    start_time=time.time()
    tokendict = Tokendict(config=config)
    csv_list = tokendict.load_csv_data(config=config)
    
    models = load_models(config, tokendict)
    if models is None:
        logger.error("Failed to load models, exiting...")
        exit(1)
    
    do_Strategy(config=config, tokendict=tokendict, models=models)
    end_time=time.time()
    print(end_time-start_time)

if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING=1
    main()

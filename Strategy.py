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


set_start_method('spawn', force=True)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_lookuptable(filename):
    """
    Load the lookup table from a CSV file.
    
    Args:
        filename (str): The name of the CSV file to load.
    
    Returns:
        DataFrame: A DataFrame containing the 'category', 'sub_category', 'org_name', and 'as' columns.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)
        
        # Select the required columns from the DataFrame
        as_org_category_subcategory = df[["category","sub_category", "org_name", "as" ]]
        
        # Log a success message
        logger.info(f"Successfully loaded lookup table from {filename}")
        
        # Return the selected columns
        return as_org_category_subcategory
    except Exception as e:
        # Log an error message if loading fails
        logger.error(f"Failed to load lookup table from {filename}: {e}")
        
        # Return None if an error occurs
        return None

def trans_into_bin(routingprefix):
    # Split the routing prefix into the IP address and subnet length
    try:
        address, length = routingprefix.split("/")
        # Convert the subnet length to an integer
        length = int(length)
        # Create an IPv6Address object from the IP address
        ipv6_object = ipaddress.IPv6Address(address)
        # Convert the IPv6 address to its binary representation
        routing_binary_representation = format(int(ipv6_object), '0128b')
        # Log the binary representation of the routing prefix
        logger.debug(f"Routing prefix {routingprefix} converted to binary: {routing_binary_representation[:length]}")
        # Return the binary representation up to the specified length
        return routing_binary_representation[:length]
    except Exception as e:
        # Log any errors that occur during the conversion process
        logger.error(f"Failed to convert routing prefix {routingprefix} to binary: {e}")
        # Return None if an error occurs
        return None
    

def do_Strategy(config, tokendict, models):
    # This function executes a strategy to generate FRPs based on a given model and configuration.
    tokenized_Dataset = []
    
    try:
        # Open the prediction file for reading
        with open(config["Prediction_path"], 'r') as file:
            logger.info(f"Opened {config['Prediction_path']} for reading")
            lines = file.readlines()
            environment = Environment(config)
            
            # Process each line in the file
            for line in lines:
                reader = csv.reader([line])
                input_token = next(reader)
                tokenized_Dataset.extend([tokenize_test_data(input_token, tokendict.vocab)]*config["num_genarate_FRP"])
            
            # Predict FRPs using the provided model
            frps = predict_FRP(config, tokendict.vocab, models[0], tokenized_Dataset,top_p=0.6,max_length=20)
            # Open the output file for appending the results
            file = open(config["genarate_FRP_path"], "a")
            for frp in frps:
                ipv6_address, bin_length = untokenize_data(frp, tokendict.vocab, logger ,environment)
                if ipv6_address != None:
                    ipv6_address = add_ipv6_colon(ipv6_address)
                    file.write(ipv6_address + "/" + str(bin_length) + "\n")
            file.close()
            
    except Exception as e:
        # Log any errors that occur during the execution
        logger.error(f"An error occurred: {e}", exc_info=True)

def load_models(config, tokendict):
    try:
        models = []
        for i in range(config["num_models"]):
            model = load_model(config, tokendict)
            model.load_state_dict(torch.load(config["model_path"]+".pth"))  
            model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) 
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

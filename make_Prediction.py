
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
    # Load the lookup table from the specified file in the configuration

    lookuptable=load_lookuptable(config["lookuptablefile"])
    
    # Open the routing prefix file for reading
    file = open(config["routingprefix_file"])

    
    # Open the prediction file for writing the results
    with open(config["Prediction_path"], 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Iterate through each line in the routing prefix file
        for line in file:
            try:
                # Strip any leading/trailing whitespace from the line
                line=line.strip()
                # Split the line into routing prefix and AS number
                routingprefix,as_num=line.split("	")
                # Find the corresponding entry in the lookup table for the AS number
                found=lookuptable[lookuptable['as'] == int(as_num)]
                # Define the token structure: 0:asnum, 1:org_name, 2:category, 3:sub_category, 4:routing_prefix, 5:prefix, 6:active_type
                if found.empty:
                    # If no entry is found, use default values and the binary representation of the routing prefix
                    input_token=[as_num,"PAD_TOKEN","PAD_TOKEN","PAD_TOKEN"]
                    input_token.append(trans_into_bin(routingprefix))
                else:
                    # If an entry is found, use its values and the binary representation of the routing prefix
                    input_token=found.iloc[0].tolist()
                    input_token.append(trans_into_bin(routingprefix))
                # Append a constant value "TCP_443" to the token list
                input_token.append("TCP_443")
                # Write the token list to the prediction file
                writer.writerow(input_token)
            except Exception as e:
                # Print the line and error message if an exception occurs
                print(line)
                print(e)
            

    # Close the opened files
    csvfile.close()
    file.close()
        
        
        


if __name__ == '__main__':

    tokendict=Tokendict(config=config)
    csv_list=tokendict.load_csv_data(config=config)
    make_Prediction(config=config,tokendict=tokendict,model=None)





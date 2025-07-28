



import ipaddress
import os
from Config import config
import pandas as pd
import pyasn



def add_new_data_to_df(existing_df, new_data):
    """
    Add new data to an existing DataFrame and return the updated DataFrame.
    
    Parameters:
    - existing_df: The existing DataFrame to which new data will be added.
    - new_data: A dictionary containing the new data to be added.
    
    Returns:
    - updated_df: The updated DataFrame.
    """
    # Create a new DataFrame from the new data
    new_df = pd.DataFrame(new_data)
    
    # Concatenate the existing DataFrame and the new DataFrame
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Return the updated DataFrame
    return updated_df



def load_lookuptable(filename):
    df = pd.read_csv(filename)
    as_org_category_subcategory=df[["as","org_name","category","sub_category"]]
    return as_org_category_subcategory


def make_Population(seed_filename,config,active_type,header_type):
    seedfile=open(seed_filename)
    # load data
    lookuptable=load_lookuptable(config["lookuptablefile"])
    asndb = pyasn.pyasn(config["pyasnfile"])

    all_df=pd.DataFrame()
    for seed in seedfile:
        address,length=seed[:-1].split("/")
        length=int(length)
        ipv6_object = ipaddress.IPv6Address(address)
        as_num,routing_prefix=asndb.lookup(address)

        if routing_prefix != None:
            # as,org_name,category,sub_category
            found=lookuptable[lookuptable['as'] == as_num]
            if not found.empty:
                prefix_binary_representation = format(int(ipv6_object), '0128b')

                routing_address,routing_prefixlen=routing_prefix.split("/")
                routing_prefixlen=int(routing_prefixlen)
                # ipv6_object = ipaddress.IPv6Address(routing_address)
                # routing_binary_representation = format(int(ipv6_object), '0128b')
            

                prefix_data = {
                    'routing_prefix': [prefix_binary_representation[:routing_prefixlen]],
                    'prefix': [prefix_binary_representation[:length]],
                    'active_type': [active_type]
                }
                prefix_df=pd.DataFrame(prefix_data)
            
                expanded_df = pd.concat([found.reset_index(drop=True), prefix_df], axis=1)
                
                all_df = add_new_data_to_df(all_df, expanded_df)



    all_df.to_csv(config["csv_file_path"], index=False,mode='a',header=header_type)
    

def view_all_type(config):
    # List all files in the directory specified in the config
    all_filenames=os.listdir(config["seedfilefolder"])
    
  
    # Iterate through each file in the directory
    for idx,all_filename in enumerate(all_filenames):
        
        # Set header_type to True for the first file, otherwise False
        if idx == 0:
            header_type=True
        else:
            header_type=False
            
        # Extract the active_type from the filename
        active_type=all_filename.split(".")[0]
        
        # Call make_Population function with the appropriate parameters
        make_Population(os.path.join(config["seedfilefolder"],all_filename),config,active_type,header_type=header_type)
    


if __name__ == '__main__':
    
    
    view_all_type(config)
    pass



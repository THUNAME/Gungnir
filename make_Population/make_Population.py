import ipaddress
import os
import pandas as pd
import pyasn
from concurrent.futures import ThreadPoolExecutor

def add_new_data_to_df(existing_df, new_data):
    """
    Appends new data to an existing DataFrame.

    Args:
        existing_df (pd.DataFrame): The original DataFrame to which new data will be added.
        new_data (list or dict): The new data to append, which can be a list of dictionaries or a dictionary.

    Returns:
        pd.DataFrame: A new DataFrame containing the combined data from existing_df and new_data.
    """
    new_df = pd.DataFrame(new_data)
    return pd.concat([existing_df, new_df], ignore_index=True)

def load_lookuptable(filename):
    """
    Loads a lookup table from a CSV file and selects specific columns.

    Args:
        filename (str): The path to the CSV file containing the lookup table.

    Returns:
        pd.DataFrame: A DataFrame containing only the columns 'as', 'org_name', 'category', and 'sub_category'.
    """
    df = pd.read_csv(filename, index_col=False)
    return df[["as", "org_name", "category", "sub_category"]]

def make_Population(seed_filename, config, active_type):
    """
    Processes a seed file to generate population data and appends it to a CSV file.

    For each seed (IPv6 prefix) in the seed file:
    1. It looks up the AS number and routing prefix using pyasn.
    2. It finds corresponding entries in the lookup table based on the AS number.
    3. It converts the IPv6 address to a binary string.
    4. It constructs rows of data including AS information, binary prefixes, and active type.
    5. It appends these rows to the specified CSV file without writing the header.

    Args:
        seed_filename (str): The path to the seed file containing IPv6 prefixes.
        config (dict): A configuration dictionary containing paths to the lookup table, pyasn database, and output CSV.
        active_type (str): A string indicating the active type, derived from the seed file name.
    """
    lookuptable = load_lookuptable(config["lookuptablefile"])
    asndb = pyasn.pyasn(config["pyasnfile"])

    result_rows = []

    with open(seed_filename) as seedfile:
        for seed in seedfile:
            seed = seed.strip()
            if not seed:
                continue

            # Split the seed into address and prefix length
            address, length = seed.split("/")
            length = int(length)

            # Create an IPv6 address object and look up AS information
            ipv6_object = ipaddress.IPv6Address(address)
            as_num, routing_prefix = asndb.lookup(address)

            # Skip if no routing prefix is found
            if routing_prefix is None:
                continue

            # Find corresponding entries in the lookup table
            found = lookuptable[lookuptable["as"] == as_num]
            if found.empty:
                continue

            # Convert the IPv6 address to a 128-bit binary string
            prefix_binary = format(int(ipv6_object), "0128b")
            routing_address, routing_prefixlen = routing_prefix.split("/")
            routing_prefixlen = int(routing_prefixlen)

            # Construct a dictionary for common row data
            row_dict = {
                "routing_prefix": prefix_binary[:routing_prefixlen],
                "prefix": prefix_binary[:length],
                "active_type": active_type
            }

            # Expand each found entry with the common row data and collect results
            for _, row in found.iterrows():
                result_rows.append({
                    "as": row["as"],
                    "org_name": row["org_name"],
                    "category": row["category"],
                    "sub_category": row["sub_category"],** row_dict
                })

    # Convert results to DataFrame and append to CSV without header
    all_df = pd.DataFrame(result_rows)
    all_df.to_csv(config["csv_file_path"], index=False, mode='a', header=False)

def view_all_type(config):
    """
    Processes all seed files in a directory using multi-threading.

    It first creates a new CSV file with the correct header, overwriting any existing file.
    Then, it uses a ThreadPoolExecutor to process each seed file concurrently.
    Each seed file is processed by the make_Population function.

    Args:
        config (dict): A configuration dictionary containing the path to the seed file folder and output CSV.
    """
    csv_file_path = config["csv_file_path"]
    
    # Create a new CSV file with headers, overwriting if it exists
    pd.DataFrame(columns=[
        "as", "org_name", "category", "sub_category",
        "routing_prefix", "prefix", "active_type"
    ]).to_csv(csv_file_path, index=False, mode='w')

    # Get all seed file names in the specified folder
    all_filenames = os.listdir(config["seedfilefolder"])
    
    # Process all files concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for filename in all_filenames:
            # Extract active_type from the filename (assuming format "active_type.txt")
            active_type = filename.split(".")[0]
            futures.append(executor.submit(
                make_Population,
                os.path.join(config["seedfilefolder"], filename),
                config,
                active_type
            ))
        # Wait for all threads to complete
        for future in futures:
            future.result()

if __name__ == '__main__':
    from Config import config  # Ensure Config.py exists and is configured correctly
    view_all_type(config)

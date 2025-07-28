



import ipaddress
import os
import pandas as pd
from Config import config
from Tools import add_new_data_to_df


class Environment:
    """Environment class to handle IPv6 prefixes and their binary representations."""
    def __init__(self,config):
        """Initialize the Environment with a config file."""
        Frp_filename=config["seedfile"]
        
        self.FRP_Area=set()  # Set to store FRP prefixes
        self.known_Area=set()  # Set to store known prefixes
        
        try:
            Frp_file=open(Frp_filename,"r")
            for line in Frp_file:
                address,length=line[:-1].split("/")
                length=int(length)
                ipv6_object = ipaddress.IPv6Address(address)
                binary_representation = format(int(ipv6_object), '0128b')
                self.FRP_Area.add(binary_representation[:length])
                self.known_Area.add(binary_representation[:length])
                
        except Exception as e:
            assert False, "Error in reading FRP file: "+str(e)

        
    
    def issubset(self,individual):
        """Check if the individual is a subset of known_Area."""
        for idx in range(len(individual)+1):
            if individual[:idx] in self.known_Area:
                return True
        return False


    def is_checked(self,individual):
        """Check if the individual is checked or is FRP."""
        return individual in self.known_Area or self.isFRP(individual)

        
    def isFRP(self,individual):
        """Check if the individual is in FRP_Area."""
        
        for idx in range(len(individual)+1):
            if individual[:idx] in self.FRP_Area:
                return True
        return False
    
    def add_FRP(self,individual):
        """Add an individual to FRP_Area and known_Area."""
        self.FRP_Area.add(individual)
        self.known_Area.add(individual)
    
    def add_known_Area(self,individual):
        """Add an individual to known_Area."""
        
        address,length=individual.split("/")
        length=int(length)
        binary_representation = format(int(ipaddress.IPv6Address(address)), '0128b')
        
        self.known_Area.add(binary_representation[:length])
        
    def is_legal(self,individual):
        """Check if the individual is legal."""
        if individual == None:
            return False
        else:
            address,length=individual.split("/")
            length=int(length)
            binary_representation = format(int(ipaddress.IPv6Address(address)), '0128b')
            
            return not self.is_checked(binary_representation[:length])
        


if __name__ == '__main__':
    environment=Environment(config)
    print(environment.FRP_Area)
    print('This is Environment.py')



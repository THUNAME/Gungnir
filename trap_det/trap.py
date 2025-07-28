import ipaddress
from itertools import islice
import secrets
import subprocess
import time
import re
from multiprocessing import Pool


def startScaning(config):
    
    cmd = (f"sudo -S zmap -i eno1 --ipv6-source-ip={config['local_ipv6']} "
            f"--ipv6-target-file={config['generated_address_path']} "
            f"-o {config['zmap_result_path']} -M icmp6_echoscan -B 1M --verbosity=0")
    echo = subprocess.Popen(['echo',config['passport']], stdout=subprocess.PIPE,)
    p = subprocess.Popen(cmd, shell=True, stdin=echo.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    hit_rate = re.findall(r"\d+\.?\d*", p.communicate()[1][-10:].decode('utf-8'))[0]
    print(f"Hit rate: {hit_rate}%")
    return hit_rate
        


def add_ipv6_colon(ipv6):
    # add : to ipv6
    ipv6_list = []
    for i in range(0,len(ipv6),4):
        ipv6_list.append(ipv6[i:i+4])
    ipv6 = ":".join(ipv6_list)
    return ipv6



def prefix_to_sixteen(prefix):
    address,length=prefix.split("/")
    length=int(length)
    ipv6_object = ipaddress.IPv6Address(address)
    binary_representation = format(int(ipv6_object), '0128b')
    
    ipv6s=[binary_representation[:length]]*min(2**(128-length),16)
    need_complete_format="0"+str(min((128-length),4))+"b"
    
    if len(ipv6s)==1:
        ipv6s[0]=hex(int(ipv6s[0], 2))[2:]
        ipv6s[0]=add_ipv6_colon(ipv6s[0])
        return ipv6s
    else:
        for i in range(len(ipv6s)):
            
            ipv6s[i]=ipv6s[i]+format(i, need_complete_format)+''.join([bin(secrets.randbelow(2))[2] for _ in range(128-length-4)])
            # print(len(ipv6s[i]),ipv6s[i])
            ipv6s[i] = hex(int(ipv6s[i], 2))[2:]
            ipv6s[i]=add_ipv6_colon(ipv6s[i])
        return ipv6s


def checkFrp(config,prefix_set):
    
    
    config['generated_address_path']=config['generated_folder_path']+str(time.time())+'.txt'
    config['zmap_result_path']=config['zmap_folder_path']+str(time.time())+'.txt'
    
    
    file=open(config['generated_address_path'],'w')
    Frp_dict=dict()

    # generate ipv6 address for each prefix
    for prefix in prefix_set:
        try:
            ipv6s=prefix_to_sixteen(prefix)
            Frp_dict[prefix]=ipv6s
        except:
            # print(f"prefix {prefix} is invalid")
            pass
    
    # write ipv6 address to file
    for prefix,ipv6s in Frp_dict.items():
        for ipv6 in ipv6s:
            file.write(ipv6+'\n')
    file.close()

    
    # start scaning
    hit_rate=startScaning(config)

    active_address_set=set()
    file=open(config['zmap_result_path'],'r')
    for line in file:
        ipv6_object=ipaddress.IPv6Address(line.strip())
        active_address_set.add(str(ipv6_object.exploded))
    file.close()


    FRP_set=set()
    for prefix,ipv6s in Frp_dict.items():
        flag=True
        for ipv6 in ipv6s:
            if ipv6 in active_address_set:
                pass
            else:
                flag=False
                break
        if flag:
            FRP_set.add(prefix)
            
    print("to be checked FRP number:",len(prefix_set))
    print("found FRP number:",len(FRP_set))
    
    return FRP_set


def split_set_into_chunks(original_set, num_chunks):
    chunk_size = len(original_set) // num_chunks
    chunks = [set() for _ in range(num_chunks)]
    iterator = iter(original_set)
    
    for i in range(num_chunks):
        chunks[i].update(islice(iterator, chunk_size))
    
    # 处理剩余的元素
    for element in iterator:
        chunks[0].add(element)
    
    return chunks

    




def make_jump_prefix(line,jump_length=4):
    address,length=line.split("/")
    length=int(length)
    length-=jump_length
    frp=address+"/"+str(length)
    return frp


def jump_outoftrap(prefix_dict,total_budget,config):
    print("to be checked ",len(prefix_dict))
    
    prefix_set=set()
    prefix2jumpprefix=dict()
    
    for prefix,trap_length in list(prefix_dict.items()):
        if trap_length>=1:
            address,length=prefix.split("/")
            length=int(length)
            length-=trap_length
            frp=address+"/"+str(length)
            prefix_set.add(frp)
            prefix2jumpprefix[prefix]=frp
        else:
            del prefix_dict[prefix]

    FRP_set = checkFrp(config, prefix_set)
    total_budget+=len(prefix_set)
    save_to_file(FRP_set)
    
    for prefix,jumpprefix in list(prefix2jumpprefix.items()):
        if jumpprefix in FRP_set:
            prefix_dict[jumpprefix]=prefix_dict[prefix]
            del prefix_dict[prefix]
        else:
            prefix_dict[prefix]/=2
            
    
    return total_budget,prefix_dict
 
   
def save_to_file(FRP_set):
    with open("50ICMPv60.6.txt", 'a') as file:
            for line in FRP_set:
                file.write(line + '\n') 

def main():
    config=dict(
        predict_path="/home/weichentian/quick_check/50ICMPv60.6.txt",
        passport="!@#weichentian$%^",
        local_ipv6="2402:f000:6:1e00::232",
        generated_folder_path="scanningdata/generated_address",
        zmap_folder_path="scanningdata/zmap_result"
        
    )
    
    # read prefix from file
    prefix_dict=dict()
    file=open(config['predict_path'],'r')
    for line in file:
        prefix_dict[line.strip()]=8
    file.close()
    
    
    # start jumping
    total_budget=0
    while True:
        total_budget,prefix_dict=jump_outoftrap(prefix_dict,total_budget,config)
        
        if len(prefix_dict)==0:
            break
        
    print("total buget:",total_budget)




    
    
if __name__ == "__main__":
    main()
    
    
    
    
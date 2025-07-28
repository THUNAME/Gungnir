
from Config import config
import pyasn


def load_black(path):
    file=open(path,"r")
    blacklist=[]
    asndb = pyasn.pyasn(config["pyasnfile"])
    for line in file:
        prefix,length=line.split("/")
        asnum,routingprefix =asndb.lookup(prefix)
        print(asnum,routingprefix)
        blacklist.append(line.strip())
    return blacklist



if __name__=="__main__":
    path="/home/weichentian/attention-frp/data/blacklist.txt"
    a=load_black(path)
    print(a)




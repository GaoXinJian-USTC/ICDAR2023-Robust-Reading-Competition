import os.path
from glob import glob
from tqdm import tqdm
import codecs
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='trocr vocab生成')
    parser.add_argument('--cust_vocab', default="./cust-data/vocab_icdar.txt", type=str, help="自定义vocab文件生成")
    parser.add_argument('--dataset_path', default=['/home/data3/gaoxinjian/icdar/seals_1W/label.txt','/home/data3/gaoxinjian/icdar/ReST_train/label.txt'], type=str, help="自定义训练数字符集")
    args = parser.parse_args()
    data_path = args.dataset_path
    vocab = set()
    
    with open("/home/data3/gaoxinjian/icdar/trocr-chinese/cust-data/chinese.txt","r") as f:
        for line in f:
            text = line.replace("\n","").split(",")[-1]
            for c in text:
                vocab.update(c)
    
    with open("/home/data3/gaoxinjian/icdar/ReST_train/chars.txt","r") as f:
        for line in f:
            text = line.replace("\n","").split(",")[-1]
            for c in text:
                vocab.update(c)
    
    if type(data_path) == list:
            for file in data_path:
                with open(file,"r") as f:
                    for line in f:
                        text = line.replace("\n","").split(",")[-1]
                        for c in text:
                            vocab.update(c)
    else:
        with open(data_path,"r") as f:
            for line in f:
                text = line.replace("\n","").split(",")[-1]
                for c in text:
                    vocab.update(c)
    
    
    root_path = os.path.split(args.cust_vocab)[0]
    os.makedirs(root_path, exist_ok=True)
    with open(args.cust_vocab, 'w') as f:
        f.write('\n'.join(list(vocab)))






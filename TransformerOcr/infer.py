import os
from PIL import Image
import numpy as np
import time
import torch
import argparse
from glob import glob
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import decode_text
from tqdm import tqdm
from datasets import load_metric
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr 模型评估')
    parser.add_argument('--cust_data_init_weights_path', default='/home/data3/gaoxinjian/icdar/trocr-chinese/cust-data/weights_', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='2', type=str, help="GPU设置")
    parser.add_argument('--data_path', default='/home/data3/gaoxinjian/icdar/ReST_train/cls_dataset/cls_train/Rectangle', type=str,
                        help="img path")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    
    # if args.random_state is not None:
    #     train_paths, test_paths = train_test_split(paths, test_size=0.05, random_state=args.random_state)

    # else:
    #     train_paths = []
    #     test_paths = paths

    
    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()

    vocab_inp = {vocab[key]: key for key in vocab}
    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    model.eval()
    model.cuda()
    img_path = args.data_path
    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}
    data = []
    data = glob.glob(f"{img_path}/*")
    pred_str, label_str = [], []
    print("test num:", len(data))
    resul_file = open("results/result.txt","w")
    for p in tqdm(data):
        path = p
        img = Image.open(f"{path}").convert('RGB')
        pixel_values = processor([img], return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values[:, :, :].cuda())
        generated_text = decode_text(generated_ids[0].cpu().numpy(), vocab, vocab_inp)
        pred_str.append(generated_text)
        name = path.split("/")[-1]
        resul_file.writelines(f"{name},{generated_text}\n") 
        print(p,'\t',generated_text)


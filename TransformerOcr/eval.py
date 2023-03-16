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

cer_metric = load_metric("./cer.py")


def compute_metrics(pred_str, label_str):
    """
    计算cer,acc
    :param pred:
    :return:
    """
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    acc = [pred == label for pred, label in zip(pred_str, label_str)]
    acc = sum(acc) / (len(acc) + 0.000001)
    return {"cer": cer, "acc": acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr 模型评估')
    parser.add_argument('--cust_data_init_weights_path', default='/home/data3/gaoxinjian/icdar/trocr-chinese/cust-data/weights_', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0', type=str, help="GPU设置")
    parser.add_argument('--test_dataset_path', default='/home/data3/gaoxinjian/icdar/ReST_train/test_.txt', type=str,
                        help="img path")
    parser.add_argument('--random_state', default=None, type=int, help="用于训练集划分的随机数")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    ann_path = args.test_dataset_path
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

    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}
    data = []
    with open(ann_path,"r") as f:
            for line in f:
                data.append(line.replace("\n",""))
    pred_str, label_str = [], []
    print("test num:", len(data))
    resul_file = open("results/result.txt","w")
    for p in tqdm(data):
        path,label = p.split(",")
        img = Image.open(f"{path}").convert('RGB')
        pixel_values = processor([img], return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values[:, :, :].cuda())
        generated_text = decode_text(generated_ids[0].cpu().numpy(), vocab, vocab_inp)
        pred_str.append(generated_text)
        label_str.append(label)
        name = path.split("/")[-1]
        resul_file.writelines(f"{name},{generated_text},{label}\n") if generated_text != label else 0
        print(generated_text,'--------',label)

    res = compute_metrics(pred_str,label_str)
    print(res)

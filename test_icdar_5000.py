# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
dit = {'0':'Rectangle ','1':"Circle ", '2':'Triangle '}

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

sys.path.append('../PaddleOCR/')
import paddle
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program
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


class Rec_rectangle(object):
    def __init__(self,cust_data_init_weights_path):
        self.processor = TrOCRProcessor.from_pretrained(cust_data_init_weights_path)
        self.vocab = self.processor.tokenizer.get_vocab()

        self.vocab_inp = {self.vocab[key]: key for key in self.vocab}
        self.model = VisionEncoderDecoderModel.from_pretrained(cust_data_init_weights_path)
        self.model.eval()
        self.model.cuda()
        logger.info("Rectangle Recognition Model is Ready!")
        
    def recognize(self,imgp):
        img = Image.open(f"{imgp}").convert('RGB')
        pixel_values = self.processor([img], return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values[:, :, :].cuda())
        generated_text = decode_text(generated_ids[0].cpu().numpy(), self.vocab, self.vocab_inp)
        return imgp,generated_text
    
    
class Rec_circle(object):
    def __init__(self,cust_data_init_weights_path,):
        self.processor = TrOCRProcessor.from_pretrained(cust_data_init_weights_path)
        self.vocab = self.processor.tokenizer.get_vocab()
        self.vocab_inp = {self.vocab[key]: key for key in self.vocab}
        self.model = VisionEncoderDecoderModel.from_pretrained(cust_data_init_weights_path)
        self.model.eval()
        self.model.cuda()
        logger.info("Circle Recognition Model is Ready!")
        
    def recognize(self,imgp):
        img = Image.open(f"{imgp}").convert('RGB')
        pixel_values = self.processor([img], return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values[:, :, :].cuda())
        generated_text = decode_text(generated_ids[0].cpu().numpy(), self.vocab, self.vocab_inp)
        return imgp,generated_text


def main():
    rec_circle_model = Rec_circle(Circle_cust_data_init_weights_path)
    rec_rectangle_model = Rec_rectangle(Rectangle_cust_data_init_weights_path)
    global_config = config['Global']
    res = open("./result_s.txt","w+")
    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)
    # build model
    model = build_model(config['Architecture'])

    load_model(config, model)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image']
        elif op_name == "SSLRotateResize":
            op[op_name]["mode"] = "test"
        transforms.append(op)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    model.eval()
    for file in sorted(get_image_file_list(config['Global']['infer_img']),key = lambda x : int(x.split("/")[-1].split(".")[0])):
        with open(file, 'rb') as f:
            img = f.read()
            data = {'image': img}
        batch = transform(data, ops)

        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)
        preds = model(images)
        post_result = post_process_class(preds)
        for rec_result in post_result:
            score = rec_result[1]
            cls = rec_result[0]
            if score > 0.88:
                if cls == '0':
                    pred_text = rec_rectangle_model.recognize(file)
                else:
                    pred_text = rec_circle_model.recognize(file)
            else:
                pred_text = rec_circle_model.recognize(file)
            res.writelines(f"{file}\t{dit[cls]}\t{pred_text[1]}\n") 
            imgname = file.split("/")[-1]
            logger.info(f"Image:{imgname}\t Type:{dit[cls]}\t Text:{pred_text[1]}")
    logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    Circle_cust_data_init_weights_path = '/home/data3/gaoxinjian/icdar/trocr-chinese/cust-data/weights_'
    Rectangle_cust_data_init_weights_path = '/home/data3/gaoxinjian/icdar/trocr-chinese/cust-data/weights_rectang'
    main()

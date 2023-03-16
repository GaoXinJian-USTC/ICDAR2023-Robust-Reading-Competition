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
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
sys.path.append('../PaddleOCR/')
import cv2
import copy
import numpy as np
import math
import time
import traceback

import tools.infer.utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read



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

logger = get_logger()

dit = {'0':'Rectangle ','1':"Circle ", '2':'Triangle '}

class TextClassifier(object):
    def __init__(self, args):
        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        self.cls_batch_num = args.cls_batch_num
        self.cls_thresh = args.cls_thresh
        postprocess_params = {
            'name': 'ClsPostProcess',
            "label_list": args.label_list,
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, _ = \
            utility.create_predictor(args, 'cls', logger)
        self.use_onnx = args.use_onnx

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        resized_image = cv2.resize(img, (imgW, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) 
        return resized_image
    
    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        cls_res = []
        batch_num = self.cls_batch_num
        elapse = 0
        img_batch = []
        for img in img_list:
            imgC, imgH, imgW = self.cls_image_shape
            resized_image = cv2.resize(img, (imgW, imgH))
            resized_image = resized_image.astype('float32')
            resized_image = resized_image.transpose((2, 0, 1)) 
            resized_image = resized_image[np.newaxis, :]
            img_batch.append(resized_image)
            
        img_batch = np.concatenate(img_batch)
        img_batch = img_batch.copy()
        self.input_tensor.copy_from_cpu(img_batch)
        self.predictor.run()
        prob_out = self.output_tensors[0].copy_to_cpu()
        self.predictor.try_shrink_memory()
        cls_result = self.postprocess_op(prob_out)
        for rno in range(len(cls_result)):
            label, score = cls_result[rno]
            cls_res.append([label, score])
        return img_list, cls_res


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


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    # image_file_list = sorted(image_file_list,key=lambda x :int( x.split("/")[-1].split('.')[0]))
    text_classifier = TextClassifier(args)
    valid_image_file_list = []
    img_list = []
    
    #######################
    # rec_circle_model = Rec_circle(args.circle_cust_data_init_weights_path)
    # rec_rectangle_model = Rec_rectangle(args.Rectangle_cust_data_init_weights_path)
    ################
    
    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        img_list, cls_res = text_classifier(img_list)
    except Exception as E:
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()
    print(cls_res)
    for ino in range(len(img_list)):
        typ,score = cls_res[ino]
        if typ == '0':
            pred_text = rec_rectangle_model.recognize(valid_image_file_list[ino])
        else:
            pred_text = rec_circle_model.recognize(valid_image_file_list[ino])
            
        logger.info("Predicts of {}, Seal Type:{}, Text:{}".format(valid_image_file_list[ino],
                                               dit[typ],pred_text[1]))
        


if __name__ == "__main__":
    main(utility.parse_args())

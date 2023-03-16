import os
import argparse
from glob import glob
from dataset import trocrDataset, trocrDataset_txt,decode_text
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import default_data_collator
from sklearn.model_selection import train_test_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric
import torchvision.transforms as transform
from data_amg import *



best_acc = 0

def compute_metrics(pred):
    """
    计算cer,acc
    :param pred:
    :return:
    """
    global best_acc
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = [decode_text(pred_id, vocab, vocab_inp) for pred_id in pred_ids]
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = [decode_text(labels_id, vocab, vocab_inp) for labels_id in labels_ids]
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    acc = [pred == label for pred, label in zip(pred_str, label_str)]
    # print([pred_str[0], label_str[0]])
    acc = sum(acc)/(len(acc)+0.000001)
    if acc > best_acc:
        trainer.save_model(os.path.join(args.checkpoint_path, f'best_acc_{round(acc*100,2)}'))
        processor.save_pretrained(os.path.join(args.checkpoint_path, f'best_acc_{round(acc*100,2)}'))
    best_acc = max(acc,best_acc)
    return {"cer": cer, "acc": acc}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune训练')
    parser.add_argument('--cust_data_init_weights_path', default='/home/data3/gaoxinjian/icdar/trocr-chinese/cust-data/weights', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--checkpoint_path', default='./checkpoint/trocr_rectangle', type=str, help="训练模型保存地址")
    parser.add_argument('--train_dataset_path', default=['/home/data4/zyh/generate_seal_2w/new_3type_2w_rectangle_data/type_1_label.txt',
                                                         '/home/data3/gaoxinjian/icdar/PaddleOCR/rectangle_label.txt','/home/data3/gaoxinjian/icdar/PaddleOCR/rectangle_label.txt','/home/data3/gaoxinjian/icdar/PaddleOCR/rectangle_label.txt','/home/data3/gaoxinjian/icdar/PaddleOCR/rectangle_label.txt',
                                                         '/home/data3/gaoxinjian/icdar/PaddleOCR/rectangle_label.txt','/home/data4/zyh/generate_seal_2w/new20w_square/sqaure_20w_labels.txt',
                                                         '/home/data4/zyh/generate_seal_2w/new_rectangle_data/new_rectangle_2w_label.txt'], type=str, help="训练数据集")
    parser.add_argument('--test_dataset_path', default='/home/data3/gaoxinjian/icdar/ReST_train/test_.txt', type=str, help="测试数据集")
    parser.add_argument('--per_device_train_batch_size', default=64, type=int, help="train batch size")
    parser.add_argument('--per_device_eval_batch_size', default=16, type=int, help="eval batch size")
    parser.add_argument('--max_target_length', default=128, type=int, help="训练文字字符数")

    parser.add_argument('--num_train_epochs', default=40, type=int, help="训练epoch数")
    parser.add_argument('--eval_steps', default=2000, type=int, help="模型评估间隔数")
    parser.add_argument('--save_steps', default=8000, type=int, help="模型保存间隔步数")

    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='1,2', type=str, help="GPU设置")

    args = parser.parse_args()
    print("train param")
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    print("loading data .................")
    train_ann_path = args.train_dataset_path
    test_ann_path = args.test_dataset_path
    # train_paths, test_paths = train_test_split(paths, test_size=0.05, random_state=10086)
    # print("train num:", len(train_paths), "test num:", len(test_paths))

    ##图像预处理
    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}
    # transformer = lambda x: x ##图像数据增强函数，可自定义
    transformer = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])

    train_dataset = trocrDataset_txt(processor=processor, max_target_length=args.max_target_length, transformer=transformer,ann_path=train_ann_path)
    transformer = lambda x: x  ##图像数据增强函数
    eval_dataset = trocrDataset_txt(processor=processor, max_target_length=args.max_target_length, transformer=transformer,ann_path=test_ann_path)

    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 25
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    cer_metric = load_metric("./cer.py")

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=4,
        fp16=False,
        output_dir=args.checkpoint_path,
        logging_steps=50,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        save_total_limit=10,
        dataloader_num_workers = 32
    )

    # seq2seq trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_path, 'last'))
    processor.save_pretrained(os.path.join(args.checkpoint_path, 'last'))






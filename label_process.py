import os
import json

data_path = '/home/data4/zyh/ReST_train/' # chars.txt train_gts.json train_images
file = open(f"{data_path}train_gts.json")
txt = json.load(file)

dst_dict = {
  "metainfo":
    {
      "dataset_type": "TextRecogDataset",
      "task_name": "textrecog",
    },
  "data_list":
    [
    ]
}

label = open(f"{data_path}label_train.txt","w",encoding='utf-8')
for k,v in list(txt.items())[:4000]:
    p = int(k.split("_")[1])
    gt = v[0]['transcription']
    label.writelines(f"{data_path}train_images/{p}.jpg,{gt}\n")
    obj = {
        "img_path": f"train_images/{p}.jpg",
        "instances":
          [
            {
              "text": f"{gt}"
            }
          ]
        }
    dst_dict['data_list'].append(obj)

js = json.dumps(dst_dict,ensure_ascii=False)
with open(f"{data_path}label_train.json", 'w', encoding='utf-8') as f_six:
    f_six.write(js)
    

dst_dict = {
  "metainfo":
    {
      "dataset_type": "TextRecogDataset",
      "task_name": "textrecog",
    },
  "data_list":
    [
    ]
}
label = open(f"{data_path}label_test.txt","w",encoding='utf-8')
for k,v in list(txt.items())[4000:]:
    p = int(k.split("_")[1])
    gt = v[0]['transcription']
    label.writelines(f"{data_path}train_images/{p}.jpg,{gt}\n")
    obj = {
        "img_path": f"train_images/{p}.jpg",
        "instances":
          [
            {
              "text": f"{gt}"
            }
          ]
        }
    dst_dict['data_list'].append(obj)

js = json.dumps(dst_dict,ensure_ascii=False)
with open(f"{data_path}label_test.json", 'w', encoding = 'utf-8') as f_six:
    f_six.write(js)
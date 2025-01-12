import os
import pandas as pd
import numpy as np
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
import copy
from tqdm import tqdm
import pdb
from collections import defaultdict


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, filename, save_pretty=False, sort_keys=False):
            with open(filename, "w") as f:
                if save_pretty:
                    f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
                else:
                    json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))

def file_exists(file_path):
    return os.path.exists(file_path)


llama_pr_path="/data1/lch/Video-LLaMA/IJCV/IJCV_Result/yc2_pre_videollama_7b_caption.json"
llama_gt_path="/data1/lch/datasets/YouCook2/yc2_instruct_3k_val.json"

max_n_sen = 10


pr_data = load_json(llama_pr_path)
gt_data = load_json(llama_gt_path)

data = {}
data['version'] = "VERSION 1.0"
results = {}

pr_list = defaultdict(list)
for d in pr_data:
    for key, value in d.items():
        pr_list[key].append(value)
# print(pr_list)


merged_dict_list = defaultdict(lambda: defaultdict(list))


for d in gt_data:
    video = d['video'] 
    for key, value in d.items():
        merged_dict_list[video][key].append(value)
# print(merged_dict_list)
gt = [dict({'video': video}, **d) for video, d in merged_dict_list.items()]
# print(result)
# print(len(result))
for i in range(len(gt)):
    vid_name = gt[i]['video'][0]
    centens = []
    for j in range(len(gt[i]['video'])):
        centen = {}
        centen['sentence']= pr_list[vid_name][j]
        centen['timestamps'] = gt[i]['timestamps'][j]
        centen['gt_sentence'] = gt[i]['QA'][j][0]['a']
        centens.append(centen)
    # print(centens)
    
    
    results[vid_name[:-4]] = centens
    # data['results'] = results
# print(results)
    
   
    # ssss
data['results'] = results
          
save_json(data,'/data1/lch/Video-LLaMA/IJCV/IJCV_Result/result/yc2_pre_videollama_7b_caption.json')
    # data.append(line)
# print(data[0],len(data))














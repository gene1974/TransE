import json
import numpy as np
import pickle
import time
import torch

# get all ent_embedding or rel_embedding
def get_embeds(embedding_layer, ent_list, emb_dim = 768):
    n_ent = len(ent_list)
    ent_ids = torch.tensor(list(range(n_ent)), dtype = int, device = 'cuda')
    ent_embeds = torch.zeros((n_ent, emb_dim))
    batch_size = 16
    begin = 0
    while begin < n_ent:
        end = min(n_ent, begin + batch_size)
        ent_embeds[begin: end] = embedding_layer(ent_ids[begin: end])
        begin += batch_size
    return ent_embeds

# load entity
def get_entity():
    data = json.load(open('all_output_multi_source_drug_processed.json')) # 171414, 384323
    entity = data['entity'] # 79756 81607 81607
    ent_name = {} # ID_2_name
    ent_dict = {} # ID_2_index
    ent_list = [] # ID
    ent_type = [] # index_2_type
    type_list = set()
    for ent in entity:
        if '名称' in ent:
            ent_name[ent['ID']] = ent['名称']
            ent_dict[ent['ID']] = len(ent_dict) # 存在一个名称对应多个ID的情况
            ent_list.append(ent['ID'])
            ent_type.append(ent['类型'][0][0][0])
            type_list.add(ent['类型'][0][0][0]) # 一个实体可能有多种类型

    type_list = list(type_list)
    type_dict = {type_list[i]: i for i in range(len(type_list))}
    return ent_dict, ent_list, ent_name, ent_type, type_dict

# clean type
def clean_type(ent_type, type_dict):
    for i in range(len(ent_type)):
        if ent_type[i] in ['实验室检查', '影像学检查', '体格检查']:
            ent_type[i] = '检查'
        elif ent_type[i] in ['症状描述', '症状', '独立症状']:
            ent_type[i] = '症状'
    type_dict.pop('实验室检查')
    type_dict.pop('影像学检查')
    type_dict.pop('体格检查')
    type_dict.pop('症状描述')
    type_dict.pop('独立症状')
    for i, key in enumerate(type_dict):
        type_dict[key] = i
    type_dict['检查'] = len(type_dict)
    return ent_type, type_dict

# get number of list
def cal_type_num(type_dict, label):
    print(type_dict)
    type_list = {type_dict[i]: i for i in type_dict}
    type_num = {key: 0 for key in type_dict}
    for i in label:
        type_num[type_list[i]] += 1
    print(type_num)

def clean_str(token):
    return token.replace('(', '').replace(')', '').replace('/', '').replace('-', '').replace('[', '').replace(']', '').replace('~', '').replace('', '')

def save_data(path, data):
    model_time = time.strftime('%m%d%H%M', time.localtime())
    with open(path + '.' + model_time, 'wb') as f:
        pickle.dump(data, f)

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


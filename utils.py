import json
import numpy as np
import pickle
import time
import torch

# embed entities
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

def clean_str(token):
    return token.replace('(', '').replace(')', '').replace('/', '').replace('-', '').replace('[', '').replace(']', '').replace('~', '').replace('', '')

def save_data(path, data):
    data_time = time.strftime('%m%d%H%M', time.localtime())
    with open(path + '.' + data_time, 'wb') as f:
        pickle.dump(data, f)
    print('Save data: ', path + '.' + data_time)
    return data_time

def load_data(path, data_time = None):
    if data_time is not None:
        path = path + '.' + data_time
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


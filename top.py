import numpy as np
import json

from utils import get_entity
from visual import do_emb, visual_emb

def cal_num(ent_dict):
    occur = {ent: 0 for ent in ent_dict}
    data = json.load(open('all_output_multi_source_drug_processed.json')) # 171414, 384323
    for line in data['relation']:
        head, tail = line[0], line[1]
        occur[head] += 1
        occur[tail] += 1
    occur = sorted([[ent, occur[ent]] for ent in occur], key = lambda x: x[1], reverse = True)
    return occur

def get_top_index(k = 500):
    ent_dict, ent_list, ent_name, ent_type, type_dict = get_entity()
    occur = cal_num(ent_dict)
    top_occur = occur[:k]
    index = [ent_dict[item[0]] for item in occur[:k]]
    return index

def visual_top(k = 500):
    index = get_top_index(k)
    
    ent_embeds, label, title, legend = do_emb('transe', '09161640', 200)
    visual_emb(ent_embeds[index], label[index], title, legend)
    ent_embeds, label, title, legend = do_emb('transe', '10222237')
    visual_emb(ent_embeds[index], label[index], 'bert-transe', legend)
    ent_embeds, label, title, legend = do_emb('bertcls')
    visual_emb(ent_embeds[index], label[index], title, legend)

if __name__ == '__main__':
    visual_top(1000)
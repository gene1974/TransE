import json

from data import get_entity
from visual import visual, visual_emb

def cal_num(ent_dict):
    occur = {ent: 0 for ent in ent_dict}
    data = json.load(open('all_output_multi_source_drug_processed.json')) # 171414, 384323
    for line in data['relation']:
        head, tail = line[0], line[1]
        occur[head] += 1
        occur[tail] += 1
    occur = sorted([[ent, occur[ent]] for ent in occur], key = lambda x: x[1], reverse = True)
    return occur

def get_top_index(ent_dict, k = 500):
    ent_dict = get_entity()
    occur = cal_num(ent_dict)
    top_occur = occur[:k]
    index = [ent_dict[item[0]] for item in occur[:k]]
    return index

def visual_top(ent_dict, ent_list, ent_name, type_dict, label, k = 500):
    index = get_top_index(ent_dict, k)
    
    visual(ent_list[index], ent_name, type_dict, label, 'transe', '09161640', 200)
    visual(ent_list[index], ent_name, type_dict, label, 'transe', '10222237')
    visual(ent_list[index], ent_name, type_dict, label, 'bertcls')

if __name__ == '__main__':
    ent_dict, ent_list, ent_name, type_dict, label = get_entity()
    visual_top(ent_dict, ent_list, ent_name, type_dict, label, 2000)
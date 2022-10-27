import json
import numpy as np

from utils import load_data, save_data

ent_type_dict = {
    '疾病': 0,
    '症状': 1,
    '手术': 2,
    '部位': 3,
    '生化指标': 4,
    '药物': 5,
    '实验室检查': 6,
    '影像学检查': 7,
    '体格检查': 8,
    '就诊科室': 9,
    '药品名': 10,
    '其他治疗': 11,
}
rel_dict = {
    '症状': 0,
    '部位': 1,
    '检查': 2,
    '并发症': 3,
    '规范化药品名称': 4,
    '科室': 5,
    '抗炎': 6,
    '止痛': 7,
    '解热': 8,
    '指标': 9,
    '疾病相关指标': 10,
    '治疗': 11,
    '抗病毒': 12,
}

# merge types
def merge_ent_type(ent_type, type_dict):
    for i in range(len(ent_type)):
        if ent_type[i] in ['症状描述', '症状', '独立症状']:
            ent_type[i] = '症状'
        elif ent_type[i] in ['手术治疗', '手术']:
            ent_type[i] = '手术'
    type_dict.pop('症状描述')
    type_dict.pop('独立症状')
    type_dict.pop('手术治疗')
    for i, key in enumerate(type_dict):
        type_dict[key] = i
    return ent_type, type_dict

# merge relation types
def merge_rel_type(rel_dict, triplets):
    for i in range(len(triplets)):
        if triplets[i][2] in ['患病症状', '症状']:
            triplets[i][2] = '症状'
        elif triplets[i][2] in ['部位', '患病部位']:
            triplets[i][2] = '部位'
    rel_dict.pop('患病症状')
    rel_dict.pop('患病部位')
    for i, key in enumerate(rel_dict):
        rel_dict[key] = i
    return rel_dict, triplets

# load and merge entity from raw data
def get_entity():
    data = json.load(open('all_output_multi_source_drug_processed.json')) # 171414, 384323
    entity = data['entity'] # 79756 81607 81607
    ent_name = {} # ID_2_name
    ent_dict = {} # ID_2_index, ID is in data, index is encoding
    ent_list = [] # index_2_ID
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

    ent_type, type_dict = merge_ent_type(ent_type, type_dict)
    label = np.array([type_dict[i] for i in ent_type])
    cal_type_num(type_dict, label)
    return ent_dict, ent_list, ent_name, ent_type, type_dict, label

# load and merge triplet from raw data
def get_triplet():
    data = json.load(open('all_output_multi_source_drug_processed.json')) # 171414, 384323
    triplets = data['relation']
    rel_dict = {} # rel_2_id
    for trip in triplets:
        if trip[2] not in rel_dict:
            rel_dict[trip[2]] = len(rel_dict)
    rel_dict, triplets = merge_rel_type(rel_dict, triplets)
    rel_list = np.array([rel_dict[t[2]] for t in triplets])
    cal_type_num(rel_dict, rel_list)
    return rel_dict, rel_list, triplets

# get number of each type of ent/rel
def cal_type_num(type_dict, label):
    print('dict:', type_dict)
    type_list = {type_dict[i]: i for i in type_dict}
    type_num = {key: 0 for key in type_dict}
    for i in label:
        type_num[type_list[i]] += 1
    print('nums:', type_num)

# dump merged entity
def dump_new_entity(ent_dict, ent_name):
    with open('./drugdata/entity2id.txt', 'w') as f:
        f.write(str(len(ent_dict)) + '\n')
        for ent_id in ent_dict:
            f.write(ent_name[ent_id] + '\t' + str(ent_dict[ent_id]) + '\n')
    return

# dump merged relation and triplets
def dump_new_relation(ent_dict, rel_dict, triplets):
    with open('./drugdata/relation2id.txt', 'w') as f:
        f.write(str(len(rel_dict)) + '\n')
        for i, rel in enumerate(rel_dict):
            f.write(rel + '\t' + str(i) + '\n')
    num_total = len(triplets)
    num_train, num_valid = int(0.7 * num_total), int(0.15 * num_total)
    print('train: ', num_train, 'valid: ', num_valid, 'test: ', num_total - num_train - num_valid)
    with open('./drugdata/train2id.txt', 'w') as f:
        f.write(str(num_train) + '\n')
        for trip in triplets[:num_train]:
            f.write(str(ent_dict[trip[0]]) + '\t' + str(ent_dict[trip[1]]) + '\t' + str(rel_dict[trip[2]]) + '\n')
    with open('./drugdata/valid2id.txt', 'w') as f:
        f.write(str(num_valid) + '\n')
        for trip in triplets[num_train: num_train + num_valid]:
            f.write(str(ent_dict[trip[0]]) + '\t' + str(ent_dict[trip[1]]) + '\t' + str(rel_dict[trip[2]]) + '\n')
    with open('./drugdata/test2id.txt', 'w') as f:
        f.write(str(num_total - num_train - num_valid) + '\n')
        for trip in triplets[num_train + num_valid:]:
            f.write(str(ent_dict[trip[0]]) + '\t' + str(ent_dict[trip[1]]) + '\t' + str(rel_dict[trip[2]]) + '\n')
    return

def dump_and_save_vocab():
    ent_dict, ent_list, ent_name, ent_type, type_dict, label = get_entity()
    rel_dict, rel_list, triplets = get_triplet()
    dump_new_entity(ent_dict, ent_name)
    dump_new_relation(ent_dict, rel_dict, triplets)
    vocab = [
        ent_dict, ent_list, ent_name, ent_type, type_dict, label, 
        rel_dict, rel_list, triplets
    ]
    data_time = save_data('./result/vocab', vocab)

def load_vocab(data_time = '10271000'):
    vocab = load_data('./result/vocab', data_time)
    return vocab

if __name__ == '__main__':
    dump_and_save_vocab()


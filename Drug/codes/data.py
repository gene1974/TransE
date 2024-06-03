import json
import numpy as np

from utils import load_data, save_data

# merge types
def merge_ent_type(ent_type_list):
    for i in range(len(ent_type_list)):
        if ent_type_list[i] in ['症状描述', '症状', '独立症状']:
            ent_type_list[i] = '症状'
        elif ent_type_list[i] in ['手术治疗', '手术']:
            ent_type_list[i] = '手术'
    return ent_type_list

# merge relation types
def merge_rel_type(triplets):
    for i in range(len(triplets)):
        if triplets[i][2] in ['患病症状', '症状']:
            triplets[i][2] = '症状'
        elif triplets[i][2] in ['部位', '患病部位']:
            triplets[i][2] = '部位'
    return triplets

# load and merge entity from raw data
def get_entity():
    ent_type_dict = {
        '疾病': 0,
        '症状': 1,
        '手术': 2,
        '部位': 3,
        '药物': 4,
        '药品名': 5,
        '生化指标': 6,
        '实验室检查': 7,
        '影像学检查': 8,
        '体格检查': 9,
        '就诊科室': 10,
        '其他治疗': 11,
    }
    data = json.load(open('all_output_multi_source_drug_processed.json')) # 171414, 384323
    entity = data['entity'] # 79756 81607 81607
    ent_name = {} # ID_2_name
    ent_dict = {} # ID_2_index, ID is in data, index is encoding
    ent_list = [] # index_2_ID
    ent_type_list = [] # index_2_type
    for ent in entity:
        if '名称' in ent:
            ent_name[ent['ID']] = ent['名称']
            ent_dict[ent['ID']] = len(ent_dict) # 存在一个名称对应多个ID的情况
            ent_list.append(ent['ID'])
            ent_type_list.append(ent['类型'][0][0][0]) # 一个实体可能有多种类型

    ent_type_list = merge_ent_type(ent_type_list)
    label = np.array([ent_type_dict[i] for i in ent_type_list])
    cal_type_num(ent_type_dict, label)
    print(len(ent_dict), len(ent_list), len(ent_name))
    return ent_dict, ent_list, ent_name, ent_type_list, ent_type_dict, label

# load and merge triplet from raw data
def get_triplet():
    rel_type_dict = {
        '症状': 0,
        '部位': 1,
        '检查': 2,
        '科室': 3,
        '指标': 4,
        '疾病相关指标': 5,
        '并发症': 6,
        '规范化药品名称': 7,
        '治疗': 8,
        '抗炎': 9,
        '抗病毒': 10,
        '止痛': 11,
        '解热': 12,
    }
    rel_type_name = ['症状', '部位', '检查', '科室', '指标', '疾病相关指标', '并发症', '规范化药品名称', '治疗', '抗炎', '抗病毒', '止痛', '解热']
    data = json.load(open('all_output_multi_source_drug_processed.json')) # 171414, 384323
    triplets = data['relation']
    triplets = merge_rel_type(triplets)
    rel_type_list = np.array([rel_type_dict[t[2]] for t in triplets])
    cal_type_num(rel_type_dict, rel_type_list)
    return rel_type_dict, rel_type_list, triplets

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
    ent_dict, ent_list, ent_name, ent_type_list, ent_type_dict, label = get_entity()
    rel_type_dict, rel_type_list, triplets = get_triplet()
    dump_new_entity(ent_dict, ent_name)
    dump_new_relation(ent_dict, rel_type_dict, triplets)
    vocab = [
        ent_dict, ent_list, ent_name, ent_type_list, ent_type_dict, label, 
        rel_type_dict, rel_type_list, triplets
    ]
    data_time = save_data('./result/vocab', vocab)
    return data_time

def load_vocab(data_time = '10271433'):
    vocab = load_data('./result/vocab', data_time)
    return vocab

if __name__ == '__main__':
    # dump_and_save_vocab()
    # vocab = load_vocab()
    # ent_dict, ent_list, ent_name, ent_type_list, ent_type_dict, label, \
    #     rel_type_dict, rel_type_list, triplets = vocab
    # print(ent_name)
    get_entity()


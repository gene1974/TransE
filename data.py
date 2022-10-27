import json
import numpy as np

from utils import load_data, save_data

# --- Archived ---

def dump_raw_data(root, entity, relation):
    with open(root + 'rawdata/entity2id.txt', 'w') as f:
        f.write(str(len(entity)) + '\n')
        for ent in entity:
            if '名称' in ent:
                f.write(ent['名称'] + '\t' + str(ent['ID']) + '\n')
            else:
                f.write(ent['内容'] + '\t' + str(ent['ID']) + '\n')
    rel_dict = {}
    with open(root + 'rawdata/train2id.txt', 'w') as f:
        f.write(str(len(relation)) + '\n')
        for rel in relation:
            if rel[2] not in rel_dict:
                rel_dict[rel[2]] = len(rel_dict)
            f.write(str(rel[0]) + '\t' + str(rel[1]) + '\t' + str(rel_dict[rel[2]]) + '\n')
    with open(root + 'rawdata/relation2id.txt', 'w') as f:
        f.write(str(len(rel_dict)) + '\n')
        for i, rel in enumerate(rel_dict):
            f.write(rel + '\t' + str(i) + '\n')
    return rel_dict

def dump_entity(root, entity):
    ent_name = {} # ID_2_name
    ent_dict = {} # ID_2_index
    ent_list = [] # ID
    for ent in entity:
        if '名称' in ent:
            ent_name[ent['ID']] = ent['名称']
            ent_dict[ent['ID']] = len(ent_dict) # 存在一个名称对应多个ID的情况
            ent_list.append(ent['ID'])
    
    print('entity: ', len(ent_dict))
    with open(root + 'entity2id.txt', 'w') as f:
        f.write(str(len(ent_dict)) + '\n')
        for i, ent_id in enumerate(ent_list):
            f.write(ent_name[ent_id] + '\t' + str(i) + '\n')
    return ent_dict, ent_list, ent_name

def dump_relation(root, triples):
    # relation
    rel_dict = {}
    for trip in triples:
        if trip[2] not in rel_dict:
            rel_dict[trip[2]] = len(rel_dict)
    print('relation: ', len(rel_dict))
    with open(root + 'relation2id.txt', 'w') as f:
        f.write(str(len(rel_dict)) + '\n')
        for i, rel in enumerate(rel_dict):
            f.write(rel + '\t' + str(i) + '\n')
    return rel_dict

def dump_triples(root, triples, ent_dict, rel_dict):
    num_total = len(triples)
    num_train, num_valid = int(0.7 * num_total), int(0.15 * num_total)
    print('train: ', num_train, 'valid: ', num_valid, 'test: ', num_total - num_train - num_valid)

    # random.shuffle(triples)
    # train
    with open(root + 'train2id.txt', 'w') as f:
        f.write(str(num_train) + '\n')
        for trip in triples[:num_train]:
            f.write(str(ent_dict[trip[0]]) + '\t' + str(ent_dict[trip[1]]) + '\t' + str(rel_dict[trip[2]]) + '\n')
    # valid
    with open(root + 'valid2id.txt', 'w') as f:
        f.write(str(num_valid) + '\n')
        for trip in triples[num_train: num_train + num_valid]:
            f.write(str(ent_dict[trip[0]]) + '\t' + str(ent_dict[trip[1]]) + '\t' + str(rel_dict[trip[2]]) + '\n')
    # test
    with open(root + 'test2id.txt', 'w') as f:
        f.write(str(num_total - num_train - num_valid) + '\n')
        for trip in triples[num_train + num_valid:]:
            f.write(str(ent_dict[trip[0]]) + '\t' + str(ent_dict[trip[1]]) + '\t' + str(rel_dict[trip[2]]) + '\n')

def load_entity(filename):
    ent_dict = {} # name2idx
    ent_list = [] # name
    with open(filename, 'r') as f:
        for line in f.readlines():
            try:
                ent_name, idx = line.strip().split('\t')
                ent_list.append(ent_name)
                ent_dict[ent_name] = int(idx)
            except:
                continue
    return ent_dict, ent_list

# --- End Archived ---

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

def load_vocab(data_time = None):
    vocab = load_data('./result/vocab', data_time)
    return vocab

if __name__ == '__main__':
    dump_and_save_vocab()


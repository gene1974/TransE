import json
import random

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

def dump_triples(root, triples, ent_dict):
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

if __name__ == '__main__':
    data = json.load(open('all_output_multi_source_drug_processed.json')) # 171414, 384323
    root = './drugdata/'
    ent_dict, ent_list, ent_name = dump_entity(root, data['entity']) # 79756 81607 81607
    rel_dict = dump_relation(root, data['relation']) # 15
    dump_triples(root, data['relation'], ent_dict)


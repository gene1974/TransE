

def load_labels(path):
    ent_list = []
    ent_dict = {} # ent -> id
    ent_label_list = []
    ent_label_dict = {} # ent -> label
    with open(path, 'r') as f:
        for line in f:
            ent, label = line.strip().split('\t')
            ent_list.append(ent)
            ent_dict[ent] = len(ent_dict)
            ent_label_list.append(label)
            ent_label_dict[ent] = label
    return ent_list, ent_dict, ent_label_list, ent_label_dict

def write_labels(ent_list, ent_label_dict, path):
    with open(path, 'w') as f:
        f.write(str(len(ent_list)) + '\n')
        for i, ent in enumerate(ent_list):
            # f.write(ent_label_dict[ent] + '\t' + str(i) + '\n')
            f.write(ent + '\t' + str(i) + '\n')

def load_triplets(path, ent_dict, rel_dict):
    triplets = []
    with open(path, 'r') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            head, relation, tail = ent_dict[head], rel_dict[relation], ent_dict[tail]
            triplets.append([head, relation, tail])
    return triplets

def write_triplets(path, triplets):
    with open(path, 'w') as f:
        f.write(str(len(triplets)) + '\n')
        for head, relation, tail in triplets:
            f.write(str(head) + ' ' + str(tail) + ' ' + str(relation) + '\n')

if __name__ == '__main__':
    # load data
    data_path = 'Wiki15k/'
    ent_list, ent_dict, ent_label_list, ent_label_dict = load_labels(data_path + 'entity2label.txt')
    rel_list, rel_dict, rel_label_list, rel_label_dict = load_labels(data_path + 'relation2label.txt')
    train_triplets = load_triplets(data_path + 'train.txt', ent_dict, rel_dict)
    valid_triplets = load_triplets(data_path + 'valid.txt', ent_dict, rel_dict)
    test_triplets = load_triplets(data_path + 'test.txt', ent_dict, rel_dict)

    # create OpenKE data
    write_labels(ent_list, ent_label_dict, data_path + 'entity2id.txt')
    write_labels(rel_list, rel_label_dict, data_path + 'relation2id.txt')
    write_triplets(data_path + 'train2id.txt', train_triplets)
    write_triplets(data_path + 'valid2id.txt', valid_triplets)
    write_triplets(data_path + 'test2id.txt', test_triplets)
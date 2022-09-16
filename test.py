import numpy as np
import torch
from torch.autograd import Variable
from OpenKE.openke.config import Trainer, Tester
from OpenKE.openke.module.model import TransE
from OpenKE.openke.data import TestDataLoader, TrainDataLoader

# get all ent_embedding or rel_embedding
def get_embeds(embedding_layer, ent_list):
    n_ent = len(ent_list)
    ent_ids = torch.tensor(list(range(n_ent)), dtype = int, device = 'cuda')
    ent_embeds = torch.zeros((n_ent, 200))
    batch_size = 16
    begin = 0
    while begin < n_ent:
        end = min(n_ent, begin + batch_size)
        ent_embeds[begin: end] = embedding_layer(ent_ids[begin: end])
        begin += batch_size
    return ent_embeds

def rank_sim(target, ent_embeds):
    n_ent = ent_embeds.shape[0]
    ent_sims = torch.zeros((n_ent, ))
    batch_size = 16
    begin = 0
    while begin < n_ent:
        end = min(n_ent, begin + batch_size)
        ent_sims[begin: end] = torch.cosine_similarity(target, ent_embeds[begin: end], dim = -1)
        begin += batch_size
    sorted_idx = sorted(range(n_ent), key = lambda x: ent_sims[x], reverse = True)
    return sorted_idx[1:11]

def rank_dist(target, ent_embeds):
    n_ent = ent_embeds.shape[0]
    ent_sims = torch.zeros((n_ent, ))
    batch_size = 16
    begin = 0
    while begin < n_ent:
        end = min(n_ent, begin + batch_size)
        ent_sims[begin: end] = torch.cdist(target, ent_embeds[begin: end], p = 2)
        begin += batch_size
    sorted_idx = sorted(range(n_ent), key = lambda x: ent_sims[x])
    return sorted_idx[1:11]

if __name__ == '__main__':
    # load data
    data_path = 'drugdata/'
    
    transe = TransE(
        ent_tot = 81607, # total number of entity
        rel_tot = 15,
        dim = 200, 
        p_norm = 1, 
        norm_flag = True
    ).cuda()
    transe.load_checkpoint('./checkpoint/transe_drug_09161640.ckpt')

    # 测试集上的链接预测结果
    test_dataloader = TestDataLoader("./drugdata/", "link", True)
    tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
    mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = True)
    print()

    # # 测试集中的前三个样本
    # # 打印模型预测的概率最高的前 10 个尾实体
    # print('Tail entity prediction: ')
    # ent_embeds = get_embeds(transe.ent_embeddings, ent_list)
    # rel_embeds = get_embeds(transe.rel_embeddings, rel_list)
    # with open('Wiki15k/test.txt', 'r') as f:
    #     test_triplets = f.readlines()
    # for j in [30, 50, 70]:
    #     head, relation, tail = test_triplets[j].strip().split('\t')
    #     print(ent_label_dict[head], ' + ', rel_label_dict[relation], ' = ', ent_label_dict[tail])
    #     head, relation, tail = ent_dict[head], rel_dict[relation], ent_dict[tail]
    #     target = ent_embeds[head] + rel_embeds[relation]
    #     sorted_idx = rank_dist(target.unsqueeze(0), ent_embeds)
    #     for i, index in enumerate(sorted_idx[:10]):
    #         print(i, ': ', ent_list[index], '\t', ent_label_list[index])
    #     print()
    
    # 使用 transe.predict 来预测
    # for j, [_, data_tail] in enumerate(test_dataloader):
    #     if j == 101:
    #         break
    #     if j not in [0, 50, 100]: # 取三个测试样本
    #         continue
    #     score = transe.predict({
    #         'batch_h': Variable(torch.from_numpy(data_tail['batch_h']).cuda()),
    #         'batch_t': Variable(torch.from_numpy(data_tail['batch_t']).cuda()),
    #         'batch_r': Variable(torch.from_numpy(data_tail['batch_r']).cuda()),
    #         'mode': data_tail['mode']
    #     }) # score 是距离，距离越小越相似
    #     sorted_idx = sorted(range(len(ent_list)), key = lambda x: score[x])
    #     print('Head: ', ent_label_list[data_tail['batch_h'].item()], ', Relatioin: ', rel_label_list[data_tail['batch_r'].item()])
    #     for i, index in enumerate(sorted_idx[:10]):
    #         print(i, ': ', ent_list[index], '\t', ent_label_list[index])
    #     print()

    # # 选取三个实体
    # # 打印和指定实体最相似的前10个实体
    # print('Similar entity prediction: ')
    # ent_embeds = get_embeds(transe, ent_list) # (15817, 200)
    # for target in [ent_dict['Q5994'], ent_dict['Q201293'], ent_dict['Q49117']]:
    #     sorted_idx = rank_sim(ent_embeds[target].unsqueeze(0), ent_embeds)
    #     print('Target: ', ent_label_list[target])
    #     for i, index in enumerate(sorted_idx):
    #         print(i, ': ', ent_list[index], '\t', ent_label_list[index])
    #     print()
        

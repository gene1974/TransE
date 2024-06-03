from OpenKE.openke.module.model import TransE

from data import load_entity
from utils import get_embeds, save_data

data_path = 'drugdata/'
model_name = './checkpoint/transe_drug_10231123.ckpt'
print('load model: ', model_name)

ent_dict, ent_list = load_entity(data_path + 'entity2id.txt')
rel_dict, rel_list = load_entity(data_path + 'relation2id.txt')

transe = TransE(
    ent_tot = 81607, # total number of entity
    rel_tot = 15,
    dim = 768, 
    p_norm = 1, 
    norm_flag = True
).cuda()
transe.load_checkpoint(model_name)

# 获取transe训练得到的所有的embedding
ent_embeds = get_embeds(transe.ent_embeddings, ent_list)
rel_embeds = get_embeds(transe.rel_embeddings, rel_list)

embeds = {
    'ent_embeds': ent_embeds.tolist(),
    'rel_embeds': rel_embeds.tolist(),
    'ent_list': ent_list,
    'rel_list': rel_list,
}

save_data('embeds', embeds)


import pickle
with open('embeds', 'rb') as f:
    data = pickle.load(f)
    print(data.keys())


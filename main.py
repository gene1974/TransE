
from utils import save_data
from train import train_transe
from bertmodel import get_entity, get_rel_emb, train_bertcls_emb

ent_dict, ent_list, ent_name, ent_type, type_dict, label = get_entity()
vocab = [ent_dict, ent_list, ent_name, ent_type, type_dict, label]
save_data('vocab', vocab)
train_transe('origin', 768)

bert_emb, rel_emb, data_time = train_bertcls_emb(ent_list, ent_name, label)
train_transe('bert', data_time)


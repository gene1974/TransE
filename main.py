
from bertmodel import train_bertcls_emb
from data import load_vocab
from train import train_transe
# from top import get_top_index, visual

ent_dict, ent_list, ent_name, ent_type, type_dict, label, \
    rel_dict, rel_list, triplets = load_vocab()

# index = get_top_index(ent_dict, 1000)

# visual(ent_list[index], ent_name, type_dict, label, 'transe-origin', '09161640')
# visual(ent_list[index], ent_name, type_dict, label, 'transe', '10222237')
# visual(ent_list[index], ent_name, type_dict, label, 'bertcls')

# train original transe
# transe, model_time, ent_emb, rel_emb = train_transe('origin', emb_dim = 768)

# train bert cls model
model, model_time, bert_emb, rel_emb, emb_time = train_bertcls_emb(ent_list, ent_name, type_dict, label, rel_list)

# train bert initial transe
transe, model_time, ent_emb, rel_emb = train_transe('bert', emb_time)



from bertmodel import train_bertcls_emb, get_bertcls_emb
from data import load_vocab
from transemodel import get_transe_embeds, train_transe
from visual import visual, visual_compare
from top import get_top_index, visual_top

ent_dict, ent_list, ent_name, ent_type_list, ent_type_dict, label, \
        rel_type_dict, rel_type_list, triplets = load_vocab()
print(len(ent_dict), len(ent_list), len(ent_name))

# # train original transe
# print('Train Origin TransE')
# transe, model_time, ent_emb, rel_emb = train_transe(ent_list, rel_type_list, mod = 'origin', emb_dim = 768)

# # train bert cls model
# print('Train Bert model and Bert-TransE')
# model, model_time, bert_emb, rel_emb, emb_time = train_bertcls_emb(ent_list, ent_name, ent_type_dict, label, rel_type_list)

# train bert initial transe
# rel_type_list = ['症状', '部位', '检查', '科室', '指标', '疾病相关指标', '并发症', '规范化药品名称', '治疗', '抗炎', '抗病毒', '止痛', '解热']
# bert_emb, rel_emb, emb_time = get_bertcls_emb(rel_type_list, ent_type_dict, '10271539')
# transe, model_time, ent_emb, rel_emb = train_transe(ent_list, rel_type_list, mod = 'bert', emb_time = emb_time)
# ent_emb, rel_emb, emb_time = get_transe_embeds('10271710', ent_list, rel_type_list, 'origin')

# index = get_top_index(ent_dict, 2000)
# visual(label, '10281442', 'transe')
# visual(label, '10272000', 'bert')
# visual(label, '10272210', 'bert-transe')

# visual_top(label, '10281442', 'transe', index)
# visual_top(label, '10272000', 'bert', index)
# visual_top(label, '10272210', 'bert-transe', index)

# visual_compare(label, '10282032', index)

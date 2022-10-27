import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from OpenKE.openke.module.model import TransE
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE

from bertmodel import get_bertcls_emb
from data import load_vocab

def translate(legend, type_dict):
    # legend = list(range(len(type_dict)))
    ent_translate_dict = {
        '疾病':     'Disease',
        '症状':     'Symptom',
        '手术':     'Surgery',
        '部位':     'Anatomy',
        '生化指标': 'Biochemical Indicator',
        '药物':     'Drug',
        '实验室检查':'Lab Exam',
        '影像学检查':'Radiographic Exam',
        '体格检查': 'Physical Exam',
        '就诊科室': 'Medical Department',
        '药品名':   'Drug Name',
        '其他治疗': 'Other treatment',
    }
    rel_translate_dict = {
        '症状': 'Disease-Symptom',
        '部位': 'Disease-Anatomy',
        '检查': 'Disease-Exam',
        '并发症': 'Complication',
        '规范化药品名称': 'Name Normalization',
        '科室': 'Disease-Department',
        '抗炎': 'Anti-inflammatory',
        '止痛': 'Pain Relief',
        '解热': 'Antipyretic',
        '指标': 'Disease-Indicator',
        '疾病相关指标': 'Exam-Indicator',
    }

# plot
def plot_embedding(data, label, title, legend):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()

    for i in range(len(legend)):
        plt.plot(data[label == i, 0], data[label == i, 1], '.')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.legend(legend)
    model_time = time.strftime('%m%d%H%M', time.localtime())
    plt.savefig('./fig/' + title + '.' + model_time + '.png', dpi = 300)
    return fig

def load_transe(ent_list, model_time = '09161640', emb_dim = 200):
    model_name = './checkpoint/transe_drug_{}.ckpt'.format(model_time)
    print('load model: ', model_name)
    
    transe = TransE(
        ent_tot = 81607, # total number of entity
        rel_tot = 15,
        dim = emb_dim, 
        p_norm = 1, 
        norm_flag = True
    ).cuda()
    transe.load_checkpoint(model_name)

    ent_embeds = get_embeds(transe.ent_embeddings, ent_list, emb_dim) # (81600, 200)
    return ent_embeds

# no fine-tune
def get_bert_emb(ent_list, ent_name):
    ent_list = [ent_name[ent] for ent in ent_list]
    tokenizer = BertTokenizer.from_pretrained('/data/pretrained/bert-base-chinese/')
    bert = BertModel.from_pretrained('/data/pretrained/bert-base-chinese/').to('cuda')

    tokens = tokenizer([ent for ent in ent_list], 
                       padding = 'max_length', truncation = True, max_length = 20, return_tensors = 'pt')
    
    bert_emb = []
    batch_size = 16
    for i in range(0, len(tokens['input_ids']), batch_size):
        token_ids = tokens['input_ids'][i: i + batch_size].to('cuda')
        mask_ids = tokens['attention_mask'][i: i + batch_size].to('cuda')
        bert_out = bert(token_ids, mask_ids)
        bert_out = bert(token_ids, mask_ids)['last_hidden_state'] # (n_batch, n_tokens, n_emb)
        # bert_out = bert(token_ids, mask_ids)['pooler_output'] # (n_batch, n_tokens, n_emb)
        bert_emb.append(bert_out.detach().cpu())
    bert_emb = torch.cat(bert_emb, dim = 0) # ([81607, 20, 768])
    bert_emb = torch.sum(bert_emb, dim = 1) # ([81607, 768])
    print('bert_emb:', bert_emb.shape)
    return bert_emb

def get_lm_emb():
    pass

def visual_emb(ent_embeds, label, title, legend):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(ent_embeds.detach().numpy())
    print('tsne:', result.shape)
    plot_embedding(result, label, title, legend)

def visual(ent_list, ent_name, type_dict, label, mod = 'transe', model_time = '09161640', emb_dim = 768):
    legend = list(range(len(type_dict)))
    if mod == 'transe':
        ent_embeds = load_transe(ent_list, model_time, emb_dim = emb_dim)
        title = 'transe'
    elif mod == 'bert':
        ent_embeds = get_bert_emb(ent_list, ent_name)
        title = 'bert-avg'
    elif mod == 'bertcls':
        ent_embeds, label = get_bertcls_emb()
        title = 'bert-cls'
    return ent_embeds, label, title, legend

if __name__ == '__main__':
    ent_dict, ent_list, ent_name, ent_type, type_dict, label, \
        rel_dict, rel_list, triplets = load_vocab()
    print(type_dict, rel_dict)
    # visual(ent_list, ent_name, type_dict, label, 'bertcls')
    # visual('transe', '10222237')



import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from OpenKE.openke.module.model import TransE
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE

from test import get_embeds
from bertmodel import get_bertcls_emb
from utils import get_entity, clean_type, cal_type_num, get_embeds, clean_str

# plot
def plot_embedding(data, label, title, legend):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()

    # label = label[:100]
    # data = data[:100]
    for i in range(len(legend)):
        plt.plot(data[label == i, 0], data[label == i, 1], '.')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.legend(legend)
    model_time = time.strftime('%m%d%H%M', time.localtime())
    plt.savefig(title + '.' + model_time + '.png', dpi = 300)
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

def visual(mod = 'transe', model_time = '09161640'):
    ent_dict, ent_list, ent_name, ent_type, type_dict = get_entity()
    ent_type, type_dict = clean_type(ent_type, type_dict)
    label = np.array([type_dict[i] for i in ent_type])
    legend = list(range(len(type_dict)))
    cal_type_num(type_dict, label)
    if mod == 'transe':
        ent_embeds = load_transe(ent_list, model_time, emb_dim = 768)
        title = 'transe'
    elif mod == 'bert':
        ent_embeds = get_bert_emb(ent_list, ent_name)
        title = 'bert-avg'
    elif mod == 'bertcls':
        ent_embeds, label = get_bertcls_emb()
        title = 'bert-cls'
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(ent_embeds.detach().numpy())
    print('tsne:', result.shape)
    plot_embedding(result, label, title, legend)

if __name__ == '__main__':
    visual('bertcls')
    # visual('transe', '10222237')



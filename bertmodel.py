import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from AttentionModel import AttentionPooling
from pytorchtools import EarlyStopping
# from visual import get_entity, clean_type

# load entity
def get_entity():
    data = json.load(open('all_output_multi_source_drug_processed.json')) # 171414, 384323
    entity = data['entity'] # 79756 81607 81607
    ent_name = {} # ID_2_name
    ent_dict = {} # ID_2_index
    ent_list = [] # ID
    ent_type = [] # index_2_type
    type_list = set()
    for ent in entity:
        if '名称' in ent:
            ent_name[ent['ID']] = ent['名称']
            ent_dict[ent['ID']] = len(ent_dict) # 存在一个名称对应多个ID的情况
            ent_list.append(ent['ID'])
            # ent_type[ent_dict[ent['ID']]] = ent['类型'][0][0][0]
            ent_type.append(ent['类型'][0][0][0])
            type_list.add(ent['类型'][0][0][0]) # 一个实体可能有多种类型

    type_list = list(type_list)
    type_dict = {type_list[i]: i for i in range(len(type_list))}
    return ent_dict, ent_list, ent_name, ent_type, type_dict

# clean type
def clean_type(ent_type, type_dict):
    for i in range(len(ent_type)):
        if ent_type[i] in ['实验室检查', '影像学检查', '体格检查']:
            ent_type[i] = '检查'
        elif ent_type[i] in ['症状描述', '症状', '独立症状']:
            ent_type[i] = '症状'
    type_dict.pop('实验室检查')
    type_dict.pop('影像学检查')
    type_dict.pop('体格检查')
    type_dict.pop('症状描述')
    type_dict.pop('独立症状')
    for i, key in enumerate(type_dict):
        type_dict[key] = i
    type_dict['检查'] = len(type_dict)
    return ent_type, type_dict

class BertCls(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('/data/pretrained/bert-base-chinese/').to('cuda')
        self.avg = AttentionPooling(768, 200)
        self.dropout = nn.Dropout(0.2)
        self.cls = nn.Linear(768, 11)
    def forward(self, token_ids, mask_ids):
        bert_out = self.bert(token_ids, mask_ids)['last_hidden_state'] # (n_batch, n_tokens, n_emb)
        bert_out = self.dropout(bert_out)
        bert_out = self.avg(bert_out)
        cls_out = self.cls(bert_out)
        return cls_out
    def embedding(self, token_ids, mask_ids):
        bert_out = self.bert(token_ids, mask_ids)['last_hidden_state'] # (n_batch, n_tokens, n_emb)
        bert_out = self.dropout(bert_out)
        bert_out = self.avg(bert_out)
        return bert_out # (n_batch, n_emb)

class BertData():
    def __init__(self, tokens, masks, label, batch_size = 16):
        self.tokens = tokens
        self.masks = masks
        self.label = torch.tensor(label, dtype = torch.long)
        self.batch_size = batch_size
    
    def __len__(self):
        return np.ceil(len(self.tokens) / self.batch_size)

    def __getitem__(self, i):
        begin = i * self.batch_size
        end = min(begin + self.batch_size, len(self.tokens))
        tokens = self.tokens[begin: end].to('cuda')
        masks = self.masks[begin: end].to('cuda')
        label = self.label[begin: end].to('cuda')
        return tokens, masks, label

def save_model(model, vocab):
    model_time = '{}'.format(time.strftime('%m%d%H%M', time.localtime()))
    model_path = './model/{}'.format(model_time)
    os.mkdir(model_path)
    torch.save(model.state_dict(), model_path + '/model_' + model_time)
    with open(model_path + '/vocab_' + model_time, 'wb') as f:
        pickle.dump(vocab, f)
    print('Save model:', model_path)

def load_model(model_time, model):
    model_path = './model/{}'.format(model_time)
    model.load_state_dict(torch.load(model_path + '/model_' + model_time))
    with open(model_path + '/vocab_' + model_time, 'rb') as f:
        vocab = pickle.load(f)
    return model, vocab

def train_bert(train_loader, valid_loader):
    model = BertCls().to('cuda')
    optimizer = optim.Adam(model.parameters(), lr = 3e-6)
    early_stopping = EarlyStopping(patience = 5, verbose = False)
    entrophy = nn.CrossEntropyLoss()
    epochs = 100
    avg_train_losses = []
    avg_valid_losses = []
    train_correct, train_total, valid_correct, valid_total = 0, 0, 0, 0
    for epoch in range(epochs):
        train_losses = []
        valid_losses = []
        model.train()
        for _, batch in enumerate(train_loader):
            if(len(batch[0]) == 0):
                break
            tokens, masks, labels = batch
            optimizer.zero_grad()
            output = model(tokens, masks) # (n_batch, n_class)
            print(output)
            loss = entrophy(output, labels)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            predict = torch.max(output, dim = 1).indices # (n_batch)
            train_correct += torch.sum(predict == labels).item()
            train_total += len(labels)
        avg_train_loss = np.average(train_losses)
        avg_train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                if(len(batch[0]) == 0):
                    break
                tokens, masks, labels = batch
                output = model(tokens, masks)
                loss = entrophy(output, labels)
                valid_losses.append(loss.item())
                predict = torch.max(output, dim = 1).indices # (n_batch, n_tokens)
                valid_correct += torch.sum(predict == labels).item()
                valid_total += len(labels)
            avg_valid_loss = np.average(valid_losses)
            avg_valid_losses.append(avg_valid_loss)
        
        print('[epoch {:d}] TrainLoss: {:.4f} DevLoss: {:.4f} TAcc: {:.3f} VAcc: {:.3f}'.format(
            epoch + 1, avg_train_loss, avg_valid_loss, train_correct / train_total, valid_correct / valid_total))
        early_stopping(avg_valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return model

def test_bert(model, test_set):
    test_correct, test_total = 0, 0
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(test_set):
            if(len(batch[0]) == 0):
                break
            tokens, masks, labels = batch
            output = model(tokens, masks)
            predict = torch.max(output, dim = 1).indices # (n_batch, n_tokens)
            test_correct += torch.sum(predict == labels).item()
            test_total += len(labels)
    print('Test: Acc: {:4f}'.format(test_correct / test_total))
    

def train_bertcls_emb():
    ent_dict, ent_list, ent_name, ent_type, type_dict = get_entity()
    ent_type, type_dict = clean_type(ent_type, type_dict)
    label = np.array([type_dict[i] for i in ent_type])
    ent_list = [ent_name[ent] for ent in ent_list]
    tokenizer = BertTokenizer.from_pretrained('/data/pretrained/bert-base-chinese/')
    tokens = tokenizer([ent for ent in ent_list], 
                       padding = 'max_length', truncation = True, max_length = 20, return_tensors = 'pt')
    
    n_train, n_dev = int(0.8 * len(tokens['input_ids'])), int(0.2 * len(tokens['input_ids']))
    train_set = BertData(tokens['input_ids'][:n_train], tokens['attention_mask'][:n_train], label[:n_train])
    valid_set = BertData(tokens['input_ids'][n_train:], tokens['attention_mask'][n_train:], label[n_train:])
    model = train_bert(train_set, valid_set)

    vocab = [
        ent_dict, ent_list, ent_name, ent_type, type_dict,
        label, ent_list, tokenizer, tokens
    ]
    save_model(model, vocab)
    test_bert(model, valid_set)

    batch_size = 16
    bert_emb = []
    for i in range(0, len(tokens['input_ids']), batch_size):
        token_ids = tokens['input_ids'][i: i + batch_size].to('cuda')
        mask_ids = tokens['attention_mask'][i: i + batch_size].to('cuda')
        bert_out = model.embedding(token_ids, mask_ids) # (n_batch, n_tokens, n_emb)
        bert_emb.append(bert_out.detach().cpu())
    bert_emb = torch.cat(bert_emb, dim = 0) # ([81607, 768])
    print('bert_emb:', bert_emb.shape)
   
    return bert_emb

# def save_model(model, vocab):
#     model_time = '{}'.format(time.strftime('%m%d%H%M', time.localtime()))
#     model_path = './model/{}'.format(model_time)
#     os.mkdir(model_path)
#     torch.save(model.state_dict(), model_path + '/model_' + model_time)
#     with open(model_path + '/vocab_' + model_time, 'wb') as f:
#         pickle.dump(vocab, f)
#     print('Save model:', model_path)

# def load_model(model_time, model):
#     model_path = './model/{}'.format(model_time)
#     model.load_state_dict(torch.load(model_path + '/model_' + model_time))
#     with open(model_path + '/vocab_' + model_time, 'rb') as f:
#         vocab = pickle.load(f)
#     return model, vocab

def get_bertcls_emb(model_time = '10221520'):
    model = BertCls().to('cuda')
    model, vocab = load_model(model_time, model)
    ent_dict, ent_list, ent_name, ent_type, type_dict, \
        label, ent_list, tokenizer, tokens = vocab
    
    # test model and vocab
    # n_train, n_dev = int(0.8 * len(tokens['input_ids'])), int(0.2 * len(tokens['input_ids']))
    # valid_set = BertData(tokens['input_ids'][n_train:], tokens['attention_mask'][n_train:], label[n_train:])
    # test_bert(model, valid_set)

    batch_size = 16
    bert_emb = []
    for i in range(0, len(tokens['input_ids']), batch_size):
        token_ids = tokens['input_ids'][i: i + batch_size].to('cuda')
        mask_ids = tokens['attention_mask'][i: i + batch_size].to('cuda')
        bert_out = model.embedding(token_ids, mask_ids) # (n_batch, n_tokens, n_emb)
        bert_emb.append(bert_out.detach().cpu())
    bert_emb = torch.cat(bert_emb, dim = 0) # ([81607, 768])
    print('bert_emb:', bert_emb.shape)

    # todo: save embedding
   
    return bert_emb, label


if __name__ == '__main__':
    load_bertcls_emb('10221520')

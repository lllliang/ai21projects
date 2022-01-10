from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple
from transformers import BertTokenizer
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from torch.optim.adamw import AdamW
from transformers import AdamW, BertConfig
from transformers import BertTokenizer, BertModel

from utils import *

def get_maxlen(sentences, tokenizer):
    max_len = 0
    max_sent = 0
    i=0
    # for sent in sentences:
    #     # 将文本分词，并添加 `[CLS]` 和 `[SEP]` 符号
    #     input_ids = tokenizer.encode(sent, add_special_tokens=True,truncation=True,max_length = 1024)
    #     max_len = max(max_len, len(input_ids))
    #     max_sent = max(max_sent,len(sent))
    #     if len(input_ids)>512:
    #         print(input_ids)
    #         print(sent)
    #         print('sent:',len(sent),len(sent.split(' ')),max_sent)
    #         # exit()
    #     i+=1
    #     if i%10000==0:
    #         print(max_len)
    sentences_clean = sentences.copy()
    for i in range(len(sentences)):
        # 将文本分词，并添加 `[CLS]` 和 `[SEP]` 符号
        sent = sentences[i]
        input_ids = tokenizer.encode(sent, add_special_tokens=True,truncation=True,max_length = 1024)
        max_len = max(max_len, len(input_ids))
        max_sent = max(max_sent,len(sent))
        if len(input_ids)>512:
            # print(input_ids)
            # print(sent)
            # print('sent:',len(sent),len(sent.split(' ')),max_sent)
            sentences_clean.pop(i)
            # exit()
        if i%10000==0:
            print(i,max_len)
    pickle.dump(sentences_clean,open('./savedata/sentences_clean.pkl','wb'))
    print('sent,sent_clea:',len(sentences),len(sentences_clean))
    return max_len

def main():
    # return
    filename = './savedata/sentences_clean.pkl'
    sentences = load_data(filename)
    # print(len(sentences))
    # exit()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print('1')
    model = BertModel.from_pretrained('bert-base-uncased')#.to(device)

    # max_len = get_maxlen(sentences, tokenizer)
    # print(max_len)
    # exit()
    input_ids = []
    attention_masks = []
    sentences_embedding = []
    i=0
    batchsize = 512
    for i in range(0,len(sentences),batchsize):
        sent = sentences[i:i+batchsize]
        inputs = tokenizer(sent, padding=True, truncation=True, return_tensors="pt")
        # encoded_sequence = inputs["input_ids"]
        # print(encoded_sequence)
        # decoded_sequence = tokenizer.decode(encoded_sequence[0])
        # print(decoded_sequence)
        # print(inputs)
        # inputs = inputs.to(device)
        outputs = model(**inputs)
        # print(outputs[0].shape,outputs[1].shape)
        # last_hidden_states = outputs.last_hidden_state
        last_hidden_states = outputs[0]
        sentence = torch.sum(last_hidden_states[0]/len(last_hidden_states[0]),0)
        sentences_embedding.extend(sentence.detach().numpy())
        # print(sentence)
        # print(sentence.shape)
        # break
        # print(outputs,last_hidden_states)
        # print(last_hidden_states.shape)
        # break
        # 将编码后的文本加入到列表  
        # input_ids.append(encoded_dict['input_ids'])
        
        # 将文本的 attention mask 也加入到 attention_masks 列表
        # attention_masks.append(encoded_dict['attention_mask'])
        i+=batchsize
        if i%10000==0:
            print(i)
    # for sent in sentences:
    #     # encoded_dict = tokenizer.encode_plus(
    #     #                     sent,                      # 输入文本
    #     #                     add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
    #     #                     max_length = max_len,           # 填充 & 截断长度
    #     #                     pad_to_max_length = True,
    #     #                     return_attention_mask = True,   # 返回 attn. masks.
    #     #                     return_tensors = 'pt',     # 返回 pytorch tensors 格式的数据
    #     #             )
    #     # print(sent)
    #     # break
    #     # tokenized_sequence = tokenizer.tokenize(sent)
    #     # print(tokenized_sequence)
    #     inputs = tokenizer(sent, padding=True, truncation=True, return_tensors="pt")
    #     # encoded_sequence = inputs["input_ids"]
    #     # print(encoded_sequence)
    #     # decoded_sequence = tokenizer.decode(encoded_sequence[0])
    #     # print(decoded_sequence)
    #     # print(inputs)
    #     # inputs = inputs.to(device)
    #     outputs = model(**inputs)
    #     # print(outputs[0].shape,outputs[1].shape)
    #     # last_hidden_states = outputs.last_hidden_state
    #     last_hidden_states = outputs[0]
    #     sentence = torch.sum(last_hidden_states[0],0)
    #     sentences_embedding.append(sentence.detach().numpy())
    #     # print(sentence)
    #     # print(sentence.shape)
    #     # break
    #     # print(outputs,last_hidden_states)
    #     # print(last_hidden_states.shape)
    #     # break
    #     # 将编码后的文本加入到列表  
    #     # input_ids.append(encoded_dict['input_ids'])
        
    #     # 将文本的 attention mask 也加入到 attention_masks 列表
    #     # attention_masks.append(encoded_dict['attention_mask'])
    #     i+=1
    #     if i%10000==0:
    #         print(i)

    with open('./savedata/bert_sentence_embedding.pkl','wb')as f:
        pickle.dump(sentences_embedding,f)
    # 将列表转换为 tensor
    # input_ids = torch.cat(input_ids, dim=0)
    # attention_masks = torch.cat(attention_masks, dim=0)
    # dataset = TensorDataset(input_ids, attention_masks)

    # batch_size = 32
    # dataloader = DataLoader(
    #         dataset,  # 训练样本
    #         sampler = RandomSampler(dataset), # 随机小批量
    #         batch_size = batch_size # 以小批量进行训练
    #     )

    

    # optim = AdamW(model.parameters(), lr=5e-5)

if __name__ == '__main__':
    main()
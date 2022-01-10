import gensim
import os,sys
import gzip
import random
import numpy as np
import json
import re
import pickle
import gensim
from gensim.models import word2vec
from gensim.models import KeyedVectors
import faiss
from string import punctuation
from transformers import AdamW, BertConfig
from transformers import BertTokenizer, BertModel
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



dir_path = './savedata'
punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'

def main():
    word2vec_model = KeyedVectors.load_word2vec_format('word2vec.model.bin', binary=True)
    file = dir_path + '/sentences_embedding_word2vec.pkl'
    # file = dir_path + '/sentences_embedding_word2vec_maxpooling.pkl'
    sentences_embedding = pickle.load(open(file,'rb'))
    sentences_embedding = np.array(sentences_embedding)
    sentences_file = dir_path + '/sentences.pkl'
    sentences = pickle.load(open(sentences_file,'rb'))
    # print(sentences_embedding.shape)
    print('AI helper system:')
    # query=input('请输入语句或单词：')
    # #按照一句话来处理
    # query = re.sub(r"[{}]+".format(punc)," ",query)
    # words = query.strip().split(' ')
    # embedding = []
    # # for word in words:
    # #     if len(word)>0:
    # #         l+=1
    # #         embedding.append(model1[word])
    # query_words_embedding = np.array([word2vec_model.word_vec(word) for word in words if len(words)>0 and word in word2vec_model])
    # query_embed = np.sum(query_words_embedding/len(query_words_embedding),0)
    # query_embed = np.array([query_embed])
    # # print(query_embed)
    # # query_word_embed = np.array([word2vec_model.word_vec(word) for word in words if len(words)>0 and word in word2vec_model])

    nb = len(sentences_embedding)
    d = len(sentences_embedding[0])
    nq=1
    # index = faiss.IndexFlatL2(d)   # build the index
    nlist = 100                       #聚类中心的个数
    k = 5
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
       # here we specify METRIC_L2, by default it performs inner-product search
    index.train(sentences_embedding)
    index.add(sentences_embedding)
    # D, I = index.search(query_embed, k)
    # for id in I[0]:
    #     print(sentences[id])
    # print(I)
    while(1):
        query=input('请输入语句或单词：')
        if query == '0':
            break
        #按照一句话来处理
        query = re.sub(r"[{}]+".format(punc)," ",query)
        words = query.strip().split(' ')
        query_words_embedding = np.array([word2vec_model.word_vec(word) for word in words if len(words)>0 and word in word2vec_model])
        query_embed = np.sum(query_words_embedding/len(query_words_embedding),0)
        query_embed = np.array([query_embed])
        D, I = index.search(query_embed, k)
        print('符合条件的前五个句子为：')
        for id in I[0]:
            print(sentences[id])
        # print(I)
    

    # print(query,type(query))

def bert_search():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-base-uncased')#.to(device)
    file = dir_path + '/bert_sentence_embedding.pkl'
    sentences_embedding = pickle.load(open(file,'rb'))
    sentences_embedding = np.array(sentences_embedding)
    sentences_file = dir_path + '/sentences_clean.pkl'
    sentences = pickle.load(open(sentences_file,'rb'))
    # print(sentences_embedding.shape)
    print('AI helper system:')
    # query=input('请输入语句或单词：')
    # #按照一句话来处理
    # query = re.sub(r"[{}]+".format(punc)," ",query)
    # words = query.strip().split(' ')
    # embedding = []
    # # for word in words:
    # #     if len(word)>0:
    # #         l+=1
    # #         embedding.append(model1[word])
    # query_words_embedding = np.array([word2vec_model.word_vec(word) for word in words if len(words)>0 and word in word2vec_model])
    # query_embed = np.sum(query_words_embedding/len(query_words_embedding),0)
    # query_embed = np.array([query_embed])
    # # print(query_embed)
    # # query_word_embed = np.array([word2vec_model.word_vec(word) for word in words if len(words)>0 and word in word2vec_model])

    nb = len(sentences_embedding)
    d = len(sentences_embedding[0])
    nq=1
    # index = faiss.IndexFlatL2(d)   # build the index
    nlist = 100                       #聚类中心的个数
    k = 5
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
       # here we specify METRIC_L2, by default it performs inner-product search
    index.train(sentences_embedding)
    index.add(sentences_embedding)
    # D, I = index.search(query_embed, k)
    # for id in I[0]:
    #     print(sentences[id])
    # print(I)
    while(1):
        query=input('请输入语句或单词：')
        if query == '0':
            break
        #按照一句话来处理
        query = re.sub(r"[{}]+".format(punc)," ",query)
        inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs[0]
        query_words_embedding = torch.sum(last_hidden_states[0],0)
        # words = query.strip().split(' ')
        # query_words_embedding = np.array([word2vec_model.word_vec(word) for word in words if len(words)>0 and word in word2vec_model])
        # query_embed = np.sum(query_words_embedding/len(query_words_embedding),0)
        query_embed = np.array([query_words_embedding.detach().numpy()])
        # print(query_embed.shape)
        D, I = index.search(query_embed, k)
        print('符合条件的前五个句子为：')
        for id in I[0]:
            print(sentences[id])
        # print(I)
    

    # print(query,type(query))

if __name__ == '__main__':
    # main()
    bert_search()
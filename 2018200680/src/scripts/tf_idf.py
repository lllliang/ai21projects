import string
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import os,sys
import gzip
import random
import numpy as np
import json
import pickle
import re
from gensim.models import word2vec

dir_path = './savedata'
sentences_file = dir_path + '/sentences.pkl'
sentences = pickle.load(open(sentences_file,'rb'))
# print(len(sentences))
# exit()
# print(sentences)
words = []
for sentence in sentences:
    sent = sentence.strip().split(' ')
    sent_word=[]
    for w in sent:
        if len(w) < 1:
            continue
        sent_word.append(w)
            # query_word += word
    words.append(sent_word)
    # for term in sent:
    #     if len(term)>0:
    #         words.append(term)

corpus = []
for word in words:
    corpus.append(' '.join(word)) 

#min_df=0, token_pattern='\w+'用于避免长度小于2的词被忽略并指定切分单词的模式
vectorizer=CountVectorizer(min_df=0, token_pattern='\w+')#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
# weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
# weight=tfidf.todense()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
tfidf_list=[]
batch_size = 10000
x = 0
for k in range(0,tfidf.shape[0],batch_size):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    print(x)
    x+=batch_size
    tfidf_part = tfidf[k:k+batch_size]
    weight=tfidf_part.toarray()
    for i in range(len(weight)):
        tfidf_dict={}
        for j in range(len(word)):
            getword = word[j]
            getvalue = weight[i][j]
            if getvalue != 0:  #去掉值为0的项
                # if tfidfdict.has_key(getword):  #更新全局TFIDF值
                #   tfidfdict[getword] += string.atof(getvalue)
                # else:
                #   tfidfdict.update({getword:getvalue})
                # tfidf_dict[getword]=getvalue
                tfidf_dict.update({getword:getvalue})
        tfidf_list.append(tfidf_dict)

output_path = dir_path
with open(output_path+'tfidf_list.pkl', 'wb') as f:
    pickle.dump(tfidf_list, f)
with open(output_path+'tfidf_list.txt', 'w') as f:
    # f.write(str(tfidf_list))
    for d in tfidf_list:
        f.write(str(d)+'\n')

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('word2vec.model.bin', binary=True)
def get_title_s1_embedding(word_ids, tfidf_dict):
    tf_idf=[tfidf_dict[str(idx)] for idx in word_ids if len(idx)>0]
    # print(product_id,word_ids)
    # print(len(self.words))
    words_embedding = [word2vec_model.word_vec(str(idx)) for idx in word_ids if len(idx)>0]
    tf_idf=np.array(tf_idf)
    words_embedding=np.array(words_embedding)
    # print(tf_idf.shape,words_embedding.shape)
    words_embedding = np.dot(tf_idf,words_embedding)
    return words_embedding
sentences_tfidf_embeddings = []
for i in range(len(words)):
    word_chars = words[i]
    title_embedding = get_title_s1_embedding(word_chars, tfidf_list[i])
    sentences_tfidf_embeddings.append(title_embedding)
with open(output_path+'sentences_tfidf_embeddings.pkl', 'wb') as f:
    pickle.dump(sentences_tfidf_embeddings, f)
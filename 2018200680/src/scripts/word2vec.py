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

# dir_path = os.getcwd() + '\savedata'
dir_path = './savedata'
sentences_file = dir_path + '/sentences.pkl'
sentences = pickle.load(open(sentences_file,'rb'))
# print(len(sentences))
# exit()
# print(sentences)
words = []
for sentence in sentences:
    sent = sentence.strip().split(' ')
    for term in sent:
        if len(term)>0:
            words.append(term)
with open(dir_path+'/sentence_word.txt', 'w', encoding='utf-8') as output:
    output.write(' '.join(words))


num_features = 200    # Word vector dimensionality
min_word_count = 1   # Minimum word count
num_workers = 16       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
corpus = word2vec.Text8Corpus(dir_path+"/sentence_word.txt")
print("!")
model = word2vec.Word2Vec(corpus, workers=num_workers, vector_size=num_features, min_count = min_word_count, window = context, sg = 1, sample = downsampling)
print('?')
model.init_sims(replace=True)
print('>')
# 保存模型，供日後使用
model.save("word2vec_model")
# print(model['replacementrefund'])

# # #可以在加载模型之后使用另外的句子来进一步训练模型
# # model = gensim.models.Word2Vec.load('/tmp/mymodel')
# # model.train(more_sentences)

model = gensim.models.Word2Vec.load('word2vec_model')
print(model)
# model['computer'] 
model.wv.save_word2vec_format('word2vec.model.bin', binary=True)
# model1 = word2vec.Word2Vec.load_word2vec_format('word2vec.model.bin', binary=True)
word2vec_model = KeyedVectors.load_word2vec_format('word2vec.model.bin', binary=True)
# print(model['computer'])
# print(model1.word_vec('citi'))
# model.wv.accuracy('word2vec.model.bin')

sentences_embedding = []
for sentence in sentences:
    # print(sentence)
    words = sentence.strip().split(' ')
    embedding = []
    # for word in words:
    #     if len(word)>0:
    #         l+=1
    #         embedding.append(model1[word])     and word in word2vec_model
    words_embedding = np.array([np.array(word2vec_model.word_vec(word)) for word in words if len(word)>0])
    sentence_embedding = np.sum(words_embedding/len(words_embedding),0)
    # sentence_embedding = np.max(words_embedding,axis=0)
    sentences_embedding.append(np.array(sentence_embedding))
sentences_embedding = np.array(sentences_embedding)
print(sentences_embedding.shape)
file = dir_path + '/sentences_embedding_word2vec_maxpooling.pkl'
with open(file,'wb') as f:
    pickle.dump(sentences_embedding,f)
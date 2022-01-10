import numpy as np
import os
from docx import Document
import re
import pickle
from string import punctuation
import sys
# print(np.random.randint(0,10,10))
# print(os.getcwd())
# dir_path = os.getcwd() + '\extract'
# save_path = os.getcwd() + '\savedata'
dir_path = './extract'
save_path = './savedata'
i=0
corpus = []
punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
print(punc)
for root, dirs, files in os.walk(dir_path):  # 获取当前文件夹的信息
    print(root)
    print(dirs)
    print(files)
    # i+=1
    # if i>1:
    #     break
    for file in files:                       # 扫描所有文件
        if os.path.splitext(file)[1] == ".docx" and file[0]!='~':# 提取出所有后缀名为md的文件
        #     i+=1
            print(root,file)
            document = Document(root+'/'+file)
            for paragraph in document.paragraphs:
                # print(paragraph.text)
                para = str(paragraph.text)
                sentences = re.split(r'[.;!?]',para.strip())
                # print(para, sentences)
                for sentence in sentences:
                    if len(sentence.split(' '))>6:
                        sentence = re.sub(r"[{}]+".format(punc)," ",sentence)
                        if len(sentence)>2 and sentence.isspace()==False:
                            corpus.append(sentence)
                #     print(sentence.split(' '))
            # print(corpus)
            # exit()
                # break
# for sentence in corpus:
    # if len(sentence)
save_file = save_path + '/sentences.pkl'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
# with open(save_file,'wb') as f:
#     pickle.dump(corpus,f)
pickle.dump(corpus,open(save_file,'wb'))

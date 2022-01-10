import pickle 
import os
import faiss
import numpy as np
from time import *
#from ClassLib import Config
from sentence_transformers import SentenceTransformer
from flask import Flask, make_response, render_template, redirect, url_for, request , send_from_directory, jsonify
from flask_apscheduler import APScheduler

init_begin_time = time()
snum = [{'name':'5'},{'name':'10'},{'name':'20'},{'name':'50'}]
model = SentenceTransformer('all-MiniLM-L6-v2')
datapath = "sentencetovdata_clean.txt"
if (os.path.exists(datapath) and os.path.isfile(datapath) and os.path.getsize(datapath) > 0):
    with open(datapath, "rb") as st2v:
        data = pickle.load(st2v)
xb = np.array(data[1])
print(len(data[0]))
d = len(xb[1])
quantizer = faiss.IndexFlatL2(d)
nlist = 100  
index = faiss.IndexIVFPQ(quantizer, d, nlist,4, 8)
index.train(xb)

print(index.is_trained)
index.add(xb) 
print(index.ntotal)
index.nprobe = 5
init_end_time = time()
init_run_time = init_end_time-init_begin_time
print("Model预处理时间:",init_run_time)
app = Flask(__name__)
#app.config.from_object(Config())  # 为实例化的flask引入配置


def fabs(a):
    if a < 0:
        return -a
    else:
        return a
@app.route("/", methods=["GET"])
def form():
    
    print("Someone link to /") 
    
    return render_template('Get.html',Selectnum = snum)
@app.route("/submit", methods=["POST"])
def form_submit():
    query_begin_time = time()
    query = request.form["name"]
    k = int(request.form["num_select"])
    print("query:",query)
    print("k: ",k)
    xq = np.random.random((1, len(xb[1]) )).astype('float32')
    query_embeddings = model.encode(query)
    for i in range(len(query_embeddings)):
        xq[0][i] = query_embeddings[i]

    D, I = index.search(xq, k) # sanity check
    print(D)
    print(I)
    Filedt = []
    for i in range(k):
        print(str(i) + data[0][ I[0][i] ])
        dis = "inf"
        if fabs(D[0][i]-0.0) > 1e-6:
            dis = str(1.0/D[0][i])
        Filedt.append({'text':data[0][I[0][i]],'rank':str(i+1),'id':str(I[0][i]),'dis':dis})

    query_end_time = time()
    query_run_time = query_end_time-query_begin_time
    print("查询时间:",query_run_time)
    return render_template('result.html',Qstring = query,Selectnum = snum,filedt = Filedt)

if __name__ == '__main__':
    #scheduler = APScheduler()
    #scheduler.init_app(app)
    #scheduler.start()
    #app.config['JSON_AS_ASCII'] = False
    app.run(port=5000)

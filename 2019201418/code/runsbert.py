from sentence_transformers import SentenceTransformer
import pickle 
import os
model = SentenceTransformer('all-MiniLM-L6-v2')
print("finish model")
if (os.path.exists("Docdata.txt") and os.path.isfile("Docdata.txt") and os.path.getsize("Docdata.txt") > 0):
    with open("Docdata.txt", "rb") as TAX:
        data = pickle.load(TAX)
print("finish read")
sentence = []
for st in data:
    for i in st:
        sentence.append(i)

sentence_embeddings = model.encode(sentence)
ans = []
ans.append(sentence)
ans.append(sentence_embeddings)
print("finish sentence_embeddings")
#for sentence, embedding in zip(sentence, sentence_embeddings):
   # print("Sentence:", sentence)
    #print("Embedding:", len(embedding))
    #print("")
with open("sentencetovdata.txt", "wb") as st2v:
    pickle.dump(ans,st2v)
print("finish all!")
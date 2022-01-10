import os
import json
import torch
import random
import numpy as np
import Levenshtein
from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import get_linear_schedule_with_warmup
from args import read_options

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class BertSim():
    def __init__(self, args) -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # init KG
        # read entities, relations and triples
        self.dataset = args.sim_dataset
        self.read_KG()
        
        # init model
        self.model_path = args.sim_model_path
        self.tokenizer  = BertTokenizer.from_pretrained(self.model_path)
        self.model      = BertForNextSentencePrediction.from_pretrained(self.model_path).to(self.device)
        
        # train params
        self.batch_size    = args.sim_batch_size
        self.learning_rate = args.sim_learning_rate 
        self.epochs        = args.sim_train_epochs
        self.neg_num       = args.sim_neg_num
        self.neg_gap       = args.sim_neg_gap
        self.optimizer     = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.save_path     = args.sim_save_path
        
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
        
        #self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.num_warmup_steps, self.num_train_steps)
    
    
    def read_KG(self):
        entity_path = os.path.join(self.dataset, "entities_name.txt")
        relation_path = os.path.join(self.dataset, "relations.txt")
        triples_path = os.path.join(self.dataset, "triples.json")
        
        self.entities, self.relations = {}, {}
        self.processed_r = []
        with open(entity_path, 'r') as f:
            entity_lines = f.readlines()
            for i,entity in enumerate(entity_lines):
                self.entities[i] = entity.strip()
        with open(relation_path, 'r') as f:
            relation_lines = f.readlines()
            for i,relation in enumerate(relation_lines):
                self.relations[i] = relation.strip()
                self.processed_r.append(" ".join(relation.strip().split(".|_")))
        self.triples = json.load(open(triples_path, 'r'))
    

    def data_process(self):
        data_path = os.path.join(self.dataset, "Simple_RQ")
        with open(data_path, "r") as f:
            pos_data = []
            for line in f:
                r,q = line.strip().split("[SEP]")
                pos_data.append([q,r,1])
        return pos_data


    def sample_neg_r(self, pos_r):
        dis = [Levenshtein.distance(pos_r, r) for r in self.processed_r]
        dis_idx = sorted(range(len(dis)), key=lambda k: dis[k])
        hard_neg = int(self.neg_num / 2)
        easy_neg = self.neg_num - hard_neg
        neg_r = self.processed_r[:hard_neg] + random.sample(self.processed_r[self.neg_gap:], easy_neg)
        return neg_r


    def next_batch_data(self, pos_data):
        all_data = []
        for q,r,l in pos_data:
            all_data.append([q,r,l])
            neg_r = self.sample_neg_r(r)
            for nr in neg_r:
                all_data.append([q,nr,0])
        random.shuffle(all_data)
        for idx in range(0,len(all_data),self.batch_size):
            yield all_data[idx:idx+self.batch_size]

            
    def train(self):
        self.model.train()

        for _ in range(self.epochs):
            pos_data = self.data_process()

            for batch_data in self.next_batch_data(pos_data):
                batch_data = np.array(batch_data)
                batch_q,batch_r,batch_l = batch_data[:,0],batch_data[:,1],batch_data[:,2]
                print(len(batch_q))
                print(batch_q)
                encode_batch = self.tokenizer(batch_q, batch_r, 
                                            padding=True, truncation=True, 
                                            return_tensors="pt").to(self.device)
                outputs = self.model(input_ids=encode_batch['input_ids'],
                                    attention_mask=encode_batch['attention_mask'],
                                    labels=torch.tensor(batch_l).unsqueeze(0))
                loss = outputs.loss
                print("loss: ", loss)
                loss.backward()
                self.optimizer.step()
                #self.scheduler.step()
        torch.save(self.model, self.save_path)


if __name__ == '__main__':
    args = read_options()
    sim_model = BertSim(args)
    sim_model.train()
    

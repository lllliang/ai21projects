def get_NER(Papers):
    entities = {}
    Relation = []
    Triples = []
    nlp = load_model('en_core_web_sm')
    model = opennre.get_model('wiki80_cnn_softmax')
    per = 0
    for idx, paper in enumerate(Papers):
        print('\r', paper, end='', flush=True)
        for sentence in Papers[paper]: 
            if len(sentence) < 20:
                continue
            res = nlp(sentence)
            ners = []
            # print(sentence)
            for ent in res.ents:
                if not check_valid(ent.text):
                    continue
                # print("( " + ent.text + ", " + ent.label_ + ")")
                temp = re.search(ent.text, sentence)
                if temp == None:
                    continue
                ners.append((ent.text, ent.label_, temp.span()))
            # ners = [(ent.text, ent.label_, re.search(ent.text, sentence).span()) for ent in res.ents]
            
            for ner in ners:  # 获得实体
                if ner[0] not in entities:
                    entities[ner[0]] = ner[1]
            
            for i in range(len(ners)):  #抽取关系
                for j in range(i + 1, len(ners)):
                    if i == j :
                        continue
                    res = model.infer({'text':sentence, 'h':{'pos':ners[i][2]}, 't':{'pos':ners[j][2]}})
                    if res[1] > 0.75:
                        Relation.append(res[0])
                        Triples.append((ners[i], res[0], ners[j])) 
        Relation = list(set(Relation))
        temp = int((float(idx)/len(Papers))*100)
        if per != temp:
            per = temp
            print('\r', str(per) + '%', end='', flush=True)



def save_data(paper):
    with open('D:\\Linux\\Three\\AI\\Paper_data.json', 'w') as wf:
        json.dump(paper, wf, indent = 4)
    print('save data successfully!')
def load_data(fp):
    with open(fp, 'r') as f:
        Paper = json.load(f)
    return Paper
def load_model(model_name):
    return spacy.load(model_name)
def check_valid(entity):
    for c in entity:
        if not ((ord(c) >= 65 and ord(c) <= 90) or (ord(c) >= 97 and ord(c) <= 122) or ord(c) == 96 or ord(c) == 46 or ord(c) == 63 or ord(c) == 33):
            return False
    return True
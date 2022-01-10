def load_model(model_name):
    return spacy.load(model_name)

def get_NER(Papers):
    entities = {}
    nlp = load_model('en_core_web_sm')
    per = 0
    for idx, paper in enumerate(Papers):
        for sentence in Papers[paper]: 
            if len(sentence) < 20:
                continue
            res = nlp(sentence)
            ners = [(ent.text, ent.label_, re.search(ent.text, sentence).span()) for ent in res.ents]
            for ner in ners:  # 获得实体
                if ner[0] not in entities:
                    entities[ner[0]] = ner[1]
            
            for i in range(ners):  #抽取关系
                for j in range(ners):
                    if i == j :
                        continue
                    
            

        temp = int((float(idx)/len(Papers))*100)
        if per != temp:
            per = temp
            print('\r', str(per) + '%', end='', flush=True)
    with open('entities.json', 'w') as wf:
        json.dump(entities, wf, indent = 4)


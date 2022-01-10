import json

def get_final_KG():
    with open('D:\\Linux\\Three\\AI\\KG\\entities_temp.json', 'r') as f1:
        entities_temp = json.load(f1)
    with open('D:\\Linux\\Three\\AI\\KG\\Triples_temp.json', 'r') as f2:
        Triples_temp = json.load(f2)
    Entities = []
    for p in entities_temp:
        Entities.append(p)
    Entities = list(set(Entities))
    
    Triples = []
    for t in Triples_temp:
        if t[0][0] == t[2][0]:
            continue
        Triples.append(t[0][0] + '\t' + t[1] + '\t' + t[2][0])
    Triples = list(set(Triples))
    
    with open('D:\\Linux\\Three\\AI\\KG\\Entities.txt', 'w') as wf1:
        for e in Entities:
            wf1.write(e + '\n')
    with open('D:\\Linux\\Three\\AI\\KG\\Triples.txt', 'w') as wf2:
        for t in Triples:
            wf2.write(t + '\n')

def get_hops(hop_num):
    with open('D:\\Linux\\Three\\AI\\KG\\Triples.json', 'r') as f:
        Triples = json.load(f)
    IN_maps = {}
    OUT_maps = {}
    for t in Triples:  # 获取出入集合
        h, r, t = t.split('\t')
        if t not in IN_maps:
            IN_maps[t] = []
        IN_maps[t].append(h)
        if h not in OUT_maps:
            OUT_maps[h] = []
        OUT_maps[h].append(t)
    return IN_maps, OUT_maps
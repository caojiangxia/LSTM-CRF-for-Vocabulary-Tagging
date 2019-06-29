#! -*- coding:utf-8 -*-

import json
from tqdm import tqdm
import codecs
import random

chars = {}
min_count = 2
train_data = []
verify_data=[]
test_data = []
chars={}
tags={}
with codecs.open('data/dataset/raw_data.txt', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        line = line.strip().split(" ")
        word = []
        tag = []
        for term in line:
            term=term.split("/")
            if len(term)==2:
                word.append(term[0])
                tag.append(term[1])
            elif len(term)==1:
                if(term[0]==""):
                    continue
                word.append(term[0])
                tag.append("w")
            elif len(term)==3:
                if term[0]=="":
                    continue
                word.append(term[0])
                tag.append(term[-1])
        choice=random.randint(1, 20)
        if choice== 1:
            test_data.append([word,tag])
            assert len(word) == len(tag)
        elif choice==2:
            verify_data.append([word,tag])
            assert len(word) == len(tag)
        else :
            train_data.append([word,tag])
            assert len(word)==len(tag)
            for w in word:
                if chars.get(w,0)==0:
                    chars[w]=0
                chars[w]+=1
            for t in tag:
                if tags.get(t, 0) == 0:
                    tags[t] = 0
                tags[t] += 1
print(len(test_data),len(verify_data),len(train_data))

with codecs.open("data/dataset/test_me.json","w",encoding="utf-8") as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)
with codecs.open("data/dataset/verify_me.json", "w", encoding="utf-8") as f:
    json.dump(verify_data, f, indent=4, ensure_ascii=False)
with codecs.open("data/dataset/train_me.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


with codecs.open('data/dataset/all_chars_me.json', 'w', encoding='utf-8') as f:
    char2id = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+2:j for i,j in enumerate(char2id)} # padding: 0, unk: 1
    char2id = {j:i for i,j in id2char.items()}
    json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)

with codecs.open('data/dataset/all_tags_me.json', 'w', encoding='utf-8') as f:
    tag2id = {i:j for i,j in tags.items()}
    id2tag = {i:j for i,j in enumerate(tag2id)}
    tag2id = {j:i for i,j in id2tag.items()}
    json.dump([id2tag, tag2id], f, indent=4, ensure_ascii=False)
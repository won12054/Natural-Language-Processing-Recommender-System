# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:51:00 2024

@author: 8778t
"""

import json
from nltk.tag import pos_tag
from sklearn_crfsuite import CRF, metrics
import random
import numpy as np

file_path = 'C:/Users/Public/6th/dataset/jungyu.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

def convert_data(datalist):
    output = []
    for data in datalist:
        sent = []
        pos = []
        for phrase in data:
            words = phrase["text"].strip().split(" ")
            while "" in words:
                words.remove("")
            if "entity" in phrase.keys():
                label = phrase["entity"]
                labels = [label + "-{}".format(i+1) for i in range(len(words))]
            else:
                labels = ["O"] * len(words)
            sent.extend(words)
            pos.extend(labels)
        output.append([sent, pos])
    return output

def sent2feats(sentence):
    feats = []
    sen_tags = pos_tag(sentence)  
    for i in range(0, len(sentence)):
        word = sentence[i]
        wordfeats = {}

        wordfeats['word'] = word        
        wordfeats['word.istitle()'] = word.istitle()
        wordfeats['word.lower()'] = word.lower()
               
        if i > 0:
            wordfeats['-1:word'] = sentence[i-1].lower()
            wordfeats['-1:pos'] = sen_tags[i-1][1]
        else:
            wordfeats['BOS'] = True
        
        if i < len(sentence) - 1:
            wordfeats['+1:word'] = sentence[i+1].lower()
            wordfeats['+1:pos'] = sen_tags[i+1][1]
        else:
            wordfeats['EOS'] = True
        
        wordfeats['pos'] = sen_tags[i][1]
        
        feats.append(wordfeats)
    return feats

def get_feats_conll(conll_data):
    feats = []
    labels = []
    for sentence in conll_data:
        feats.append(sent2feats(sentence[0]))
        labels.append(sentence[1])
    return feats, labels

train_datafile = [i["data"] for i in data["BookRestaurant"]]

train_data = convert_data(train_datafile)

split_index = int(0.8 * len(train_data))
train_data_split = train_data[:split_index]
test_data_split = train_data[split_index:]

train_feats, train_labels = get_feats_conll(train_data_split)
test_feats, test_labels = get_feats_conll(test_data_split)

def train_seq(X_train, Y_train, X_test, Y_test):
    random.seed(21)
    np.random.seed(21)
    
    crf_jungyu = CRF(algorithm='lbfgs', c1=0.001, c2=3, max_iterations=70)  
    crf_jungyu.fit(X_train, Y_train)
    
    labels = list(crf_jungyu.classes_)
    y_pred = crf_jungyu.predict(X_test)
    
    accuracy = metrics.sequence_accuracy_score(Y_test, y_pred)
    print("Sequence Accuracy:", accuracy)
    
    return crf_jungyu, y_pred

crf_jungyu, y_pred = train_seq(train_feats, train_labels, test_feats, test_labels)

print("\n16th Test Instance Prediction vs Actual:")
print("Predicted Labels:", y_pred[15])
print("Actual Labels:", test_labels[15])

new_instance = ['I', 'want', 'to', 'book', 'for', 'five', 'people', 'a', 'table', 'at', 'Le Cinq resturant', 'in', 'Paris']
new_feats = sent2feats(new_instance)
new_pred = crf_jungyu.predict([new_feats])
print("\nNew Instance Prediction:", new_pred[0])

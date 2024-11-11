# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:54:27 2024

@author: 8778t
"""
import json

'''
3b. Write code to read your json file, 
and store the elements into a group of lists.
'''
file_path = 'C:/Users/Public/6th/NLP and Recommender Systems/Assignment2/jungyu_intents.json'

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    
tags = []
patterns = []
responses = []
all_intents = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        all_intents.append((pattern, intent['tag']))  
        patterns.append(pattern)
        tags.append(intent['tag'])  
    responses.append(intent['responses'])

print(len(patterns))
print(len(tags))

'''
4. Preprocessing
4-a. 
Encode the list of intents, 
these will be the classes 
for your model in total 12 classes.
'''
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_tags = label_encoder.fit_transform(tags)

print(tags)
print(encoded_tags)

'''
4-b. Use keras tokenizer to tokenize the patterns, 
then carryout the necessary pre-processing steps 
that you think are required to prepare the data 
for a deep learning training on the intent patterns 
in order to identify intents.
'''
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=2000)  
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, maxlen=400)

print(tokenizer.word_index)
print(len(padded_sequences))
print(patterns[1])
print(tags[1])
print(padded_sequences[1])

'''
5. Deep learning training
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import Input

model = Sequential()
model.add(Input(shape=(400,))) 
model.add(Embedding(input_dim=2000, output_dim=100))
model.add(GlobalAveragePooling1D())
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(len(tags), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(padded_sequences, encoded_tags, epochs=500)


history_1000 = model.fit(padded_sequences, encoded_tags, epochs=1000)

'''
6. Testing your bot.
'''
import pickle

with open('C:/Users/Public/6th/NLP and Recommender Systems/Assignment2/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('C:/Users/Public/6th/NLP and Recommender Systems/Assignment2/label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.save('C:/Users/Public/6th/NLP and Recommender Systems/Assignment2/model.keras')






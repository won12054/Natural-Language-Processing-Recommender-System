# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:33:47 2024

@author: 8778t
"""
import json
import pickle
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

file_path = 'C:/Users/Public/6th/NLP and Recommender Systems/Assignment2/jungyu_intents.json'

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    
model = load_model('C:/Users/Public/6th/NLP and Recommender Systems/Assignment2/model.keras')

with open('C:/Users/Public/6th/NLP and Recommender Systems/Assignment2/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('C:/Users/Public/6th/NLP and Recommender Systems/Assignment2/label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

import random

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    sequence = tokenizer.texts_to_sequences([user_input])
    print(f"Tokenized input: {sequence}")  

    padded_sequence = pad_sequences(sequence, maxlen=400)
    print(f"Padded sequence: {padded_sequence}")  

    prediction = model.predict(padded_sequence)
    print(f"Raw prediction: {prediction}")  

    predicted_tag = label_encoder.inverse_transform([np.argmax(prediction)])
    print(f"Predicted tag: {predicted_tag[0]}")  

    for intent in data['intents']:
        if intent['tag'] == predicted_tag[0]:
            response = random.choice(intent['responses'])
            print(f"Bot: {response}")
            break






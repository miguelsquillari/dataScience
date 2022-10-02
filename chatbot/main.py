import random
import json
import pickle
from re import S
from unittest import result
from mysqlx import Result
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

wl = WordNetLemmatizer()
intents = json.loads(open('chat-dic.json').read())

words = pickle.load(open('words.pkl', 'rb'))
clases = pickle.load(open('clases.pkl', 'rb'))


model = load_model('chat-bot.model')

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [wl.lemmatize(word)  for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word==w:
                bag[i] = 1

    return np.array(bag)           


def predict(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    print("predition :: ",res)
    ERROR_THRESHOLD = 0.25
    result = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x:x[1], reverse=True)
    result_list = []
    for r in result:
        result_list.append({'intents':clases[r[0]],'probability':str(r[1])})
    return result_list


def get_response(intent_list, intent_json):
    tag = intent_list[0]['intents']
    list_intents = intent_json['intents']
    for i in list_intents:
        if i['tag'] ==tag:
            result = random.choice(i['responses'])
            break
    return result

print("chat-bot - MIKE")

while True:
    men = input("")
    print("mensaje a predecir :: ",men)
    intent = predict(men)
    print("imprimio :: ",intent)
    res = get_response(intent, intents)
    print(res)


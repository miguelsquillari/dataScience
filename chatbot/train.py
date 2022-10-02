from random import random, shuffle
from tabnanny import verbose
import numpy as np
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

wl = WordNetLemmatizer()
#nltk.download()
words = []
classes = []
documents = []

ignore_letters = ["?", ",", ".", "!"]

chat_dics = json.loads(open('chat-dic.json').read())


for dic in chat_dics['intents']:    
    #print("intents:",dic)
    for pattern in dic['patterns']:
        word_list = nltk.word_tokenize(pattern, language="spanish")
        words.extend(word_list)
        documents.append((word_list, dic['tag']))
        if dic['tag'] not in classes:
            classes.append(dic['tag'])


words = [wl.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

print("=====================================================================================================================")
print(words)

print("=====================================================================================================================")

print("=====================================================================================================================")
print(documents)

print("=====================================================================================================================")


pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('clases.pkl', 'wb'))


# trainning

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [wl.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    #print(bag)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])


print("training-->",training)
print("end -->")
shuffle(training)
training = np.array(training) 

train_x = list(training[:,0])
train_y = list(training[:,1])

print("len de ", (len(train_x[0]),))
model = Sequential()
model.add(Dense(128,   activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chat-bot.model')
print("finish")
 


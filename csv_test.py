import pandas as pd
import numpy as np
import tensorflow as tf
import os
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
import random

dataset = pd.read_csv(os.path.join('data', 'Conversation.csv'), index_col=False)

dataset = dataset.drop(['Unnamed: 0'], axis=1)

questions = dataset.question
answers = dataset.answer

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignoreLetter = ['?', '!', '@', '.', ',']

for index, question in enumerate(questions):
    wordList = nltk.word_tokenize(question)
    words.extend(wordList)
    documents.append((wordList, index))
    if index not in classes:
        classes.append(index)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetter ]

words = sorted(set(words))

pickle.dump(words, open('csv_test_word.pkl', 'wb'))
pickle.dump(classes, open('csv_test_classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPattern = document[0]
    wordPattern = [lemmatizer.lemmatize(word.lower()) for word in wordPattern]
    for word in words:
        bag.append(1) if word in wordPattern else bag.append(0)
    
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
hist = model.fit(np.array(trainX), np.array(trainY), epochs=300, batch_size=16, verbose=1, validation_split=0.2, shuffle=True)
model.save('csv_chatbot_model.h5', hist)
print('Done')
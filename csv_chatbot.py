from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import os
import pandas as pd
import pickle
import nltk
import numpy as np
from sortDataset import documents
import streamlit as st

lemmatizer = WordNetLemmatizer()

dataset = pd.read_csv(os.path.join('data', 'Conversation.csv'), index_col=False)

dataset = dataset.drop(['Unnamed: 0'], axis=1)

questions = dataset.question
answers = dataset.answer

words = pickle.load(open('csv_test_word.pkl', 'rb'))
classes = pickle.load(open('csv_test_classes.pkl', 'rb'))
model = load_model('csv_chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents):
    if intents:
        max_probability_item = max(intents, key=lambda x: float(x['probability']))
        return dataset['answer'][max_probability_item['intent']]
    return "Im sorry but i don't understand you"


# message = input("you : ")

# if message:
#     result = predict_class(message)
#     response = get_response(result)
#     print(response)

def send_message():
    user_message = st.session_state.user_message_input.strip()
    st.session_state.chat_history = []
    if user_message:
        # Proses pesan menggunakan model chatbot Anda
        ints = predict_class(user_message)
        res = get_response(ints)
        # Menambahkan pesan pengguna dan respons bot ke dalam chat history
        st.session_state.chat_history.append(f"You: {user_message}")
        st.session_state.chat_history.append(f"Bot: {res}")
        # Mengosongkan input box setelah pesan dikirim
        st.session_state.user_message_input = ""

# Membuat input box dengan tombol send
user_input = st.text_input("You:", key="user_message_input", on_change=send_message)

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

    
# Menampilkan chat history
for message in st.session_state.chat_history:
    st.text(message)

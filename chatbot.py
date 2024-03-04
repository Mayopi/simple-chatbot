import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
import time
import uuid

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

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

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Inisialisasi chat history dalam session state jika belum ada


# Fungsi untuk menangani pengiriman pesan
def send_message():
    user_message = st.session_state.user_message_input.strip()
    if user_message:
        if user_message.lower() == "quit":
            st.session_state.chat_history.append("You: Thank you for chatting. Goodbye!")
            # Opsional: Anda bisa menambahkan logika untuk mengakhiri sesi chat di sini
        else:
            # Proses pesan menggunakan model chatbot Anda
            ints = predict_class(user_message)
            res = get_response(ints, intents)
            # Menambahkan pesan pengguna dan respons bot ke dalam chat history
            st.session_state.chat_history.append(f"You: {user_message}")
            st.session_state.chat_history.append(f"Bot: {res}")
        # Mengosongkan input box setelah pesan dikirim
        st.session_state.user_message_input = ""

# Membuat input box dengan tombol send
user_input = st.text_input("You:", key="user_message_input", on_change=send_message)

if 'chat_history' not in st.session_state:
    print('do you even get in this block motherfucker')
    st.session_state['chat_history'] = []

    
# Menampilkan chat history
for message in st.session_state.chat_history:
    st.text(message)

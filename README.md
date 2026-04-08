RNN app.py

# streamlit run app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Load RNN model (changed file name)
model = tf.keras.models.load_model("rnn_text_generator.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max sequence length
with open("max_seq_len.pkl", "rb") as f:
    max_seq_len = pickle.load(f)

# Temperature sampling
def sample_with_temperature(preds, temperature=0.7):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# Generate text
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

        predicted = model.predict(token_list, verbose=0)[0]
        predicted_index = sample_with_temperature(predicted, temperature=0.7)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text

# UI
st.set_page_config(page_title="RNN Text Generator", layout="centered")

st.title("🧠 RNN Text Generator")
st.write("Generate text using Simple RNN model (Observe limitations)")

seed = st.text_input("Enter starting text:", "machine learning")
num_words = st.slider("Number of words to generate:", 1, 30, 10)

if st.button("Generate"):
    result = generate_text(seed, num_words)
    st.success(result);

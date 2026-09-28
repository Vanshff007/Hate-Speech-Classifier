import pickle
import re

import contractions
import nltk
import pandas as pd
import streamlit as st
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "model_TEST.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 200  # must match maxlen used in training
LABELS = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]


@st.cache_resource
def load_resources():
    for pkg in ["stopwords", "punkt", "punkt_tab"]:
        nltk.download(pkg, quiet=True)
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return model, tokenizer, stop_words


# Preprocessing must stay identical to the training notebook,
# otherwise the model sees text it was not trained on.
def processing(text, stop_words):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    words = [w.lower() for w in word_tokenize(text) if w.lower() not in stop_words]
    text = " ".join(words)
    text = contractions.fix(text)
    text = re.sub(r"[^a-zA-Z0-9' ]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"user(?:name)?\s", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text


def classify(text, model, tokenizer, stop_words):
    clean = processing(text, stop_words)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return model.predict(padded, verbose=0)[0]


def main():
    st.title("VirtuShield - Hate Speech Classifier")
    model, tokenizer, stop_words = load_resources()

    user_input = st.text_area("Enter text for classification:")

    if st.button("Classify"):
        if not user_input.strip():
            st.warning("Please enter some text for classification.")
            return
        probs = classify(user_input, model, tokenizer, stop_words)
        result = pd.DataFrame({"Class": LABELS, "Probability": probs})
        st.bar_chart(result.set_index("Class"))
        st.dataframe(result.style.format({"Probability": "{:.2%}"}), hide_index=True)


if __name__ == "__main__":
    main()

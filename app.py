import pickle

import altair as alt
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import get_stop_words, processing

MODEL_PATH = "model_TEST.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 200  # must match maxlen used in training
LABELS = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
THRESHOLD = 0.5  # a label counts as present above this probability


@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer, get_stop_words()


def classify(text, model, tokenizer, stop_words):
    clean = processing(text, stop_words)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return model.predict(padded, verbose=0)[0]


def is_toxic(probs):
    return max(probs) >= THRESHOLD


def main():
    st.title("VirtuShield - Hate Speech Classifier")
    model, tokenizer, stop_words = load_resources()

    user_input = st.text_area("Enter text for classification:")

    if st.button("Classify"):
        if not user_input.strip():
            st.warning("Please enter some text for classification.")
            return
        probs = classify(user_input, model, tokenizer, stop_words)
        if is_toxic(probs):
            found = [label for label, p in zip(LABELS, probs) if p >= THRESHOLD]
            st.error(f"Toxic ({', '.join(found)})")
        else:
            st.success("Not toxic")

        result = pd.DataFrame({"Class": LABELS, "Probability": probs})
        # Fixed 0-100% scale, so small scores do not look like big bars.
        chart = alt.Chart(result).mark_bar().encode(
            x=alt.X("Class", sort=None),
            y=alt.Y(
                "Probability",
                scale=alt.Scale(domain=[0, 1], nice=False, clamp=True),
                axis=alt.Axis(format="%", values=[0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            ),
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(result.style.format({"Probability": "{:.2%}"}), hide_index=True)


if __name__ == "__main__":
    main()

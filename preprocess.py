"""Text preprocessing shared by the app, the fine-tuning script and the tests.

The model only understands text cleaned exactly like this, so every place that
feeds text to the model must use this module.
"""
import re

import contractions
import nltk
from nltk.tokenize import word_tokenize

# Negations flip the meaning of a sentence ("you are not stupid"),
# so they must not be removed together with the other stopwords.
NEGATIONS = {"not", "no", "nor", "never"}


def get_stop_words():
    for pkg in ["stopwords", "punkt", "punkt_tab"]:
        nltk.download(pkg, quiet=True)
    return set(nltk.corpus.stopwords.words("english")) - NEGATIONS


def processing(text, stop_words):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Expand contractions before removing stopwords, so "ur" -> "you are"
    # and "don't" -> "do not" are handled like the full words.
    text = contractions.fix(text)
    words = [w.lower() for w in word_tokenize(text) if w.lower() not in stop_words]
    text = " ".join(words)
    text = re.sub(r"[^a-zA-Z0-9' ]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"user(?:name)?\s", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text.strip()

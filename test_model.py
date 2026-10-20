"""Run the model on the 500 labelled cases in test_cases.py.

Run with:  python test_model.py            (summary + failures)
           python test_model.py --all      (also print every case)
"""
import sys
from collections import defaultdict

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app import MAX_LEN, THRESHOLD, load_resources
from preprocess import processing
from test_cases import CASES


def predict(texts, model, tokenizer, stop_words):
    clean = [processing(t, stop_words) for t in texts]
    seqs = tokenizer.texts_to_sequences(clean)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    return model.predict(padded, batch_size=256, verbose=0)


def evaluate(model, tokenizer, stop_words, show_all=False):
    probs = predict([text for text, _, _ in CASES], model, tokenizer, stop_words)
    scores = probs.max(axis=1)

    per_category = defaultdict(lambda: [0, 0])
    failures = []
    for (text, expected, category), score in zip(CASES, scores):
        predicted = bool(score >= THRESHOLD)
        per_category[category][0] += predicted == expected
        per_category[category][1] += 1
        if predicted != expected:
            failures.append((category, text, expected, score))
        if show_all:
            mark = "ok " if predicted == expected else "BAD"
            print(f"{mark} [{category}] {score:6.1%}  {text}")

    print(f"\n{'category':<18}{'correct':>10}")
    for category, (ok, total) in per_category.items():
        print(f"{category:<18}{ok:>5}/{total:<4}")
    correct = sum(ok for ok, _ in per_category.values())
    print(f"{'TOTAL':<18}{correct:>5}/{len(CASES)}  ({correct / len(CASES):.1%})")

    if failures:
        print(f"\n{len(failures)} failures (score = highest label probability):")
        for category, text, expected, score in failures:
            want = "toxic" if expected else "clean"
            print(f"  [{category}] want {want}, got {score:.1%}: {text}")
    return correct / len(CASES)


if __name__ == "__main__":
    model, tokenizer, stop_words = load_resources()
    accuracy = evaluate(model, tokenizer, stop_words, show_all="--all" in sys.argv)
    sys.exit(0 if accuracy >= 0.95 else 1)

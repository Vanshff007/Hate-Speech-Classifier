"""Quick check that the model loads and gives sensible predictions.

Run with:  python test_model.py
"""
from app import LABELS, classify, load_resources

SAMPLES = {
    "clean": [
        "Thank you for your help, have a great day!",
        "I think this article needs more sources.",
    ],
    "toxic": [
        "You are a stupid idiot and nobody likes you.",
        "Shut up you worthless piece of garbage.",
    ],
}

model, tokenizer, stop_words = load_resources()
print("Model and tokenizer loaded.\n")

scores = {}
for group, texts in SAMPLES.items():
    scores[group] = []
    for text in texts:
        probs = classify(text, model, tokenizer, stop_words)
        assert len(probs) == 6, "model should return 6 probabilities"
        assert all(0 <= p <= 1 for p in probs), "probabilities must be between 0 and 1"
        scores[group].append(probs[0])  # "Toxic" score
        print(f"[{group}] {text}")
        for label, p in zip(LABELS, probs):
            print(f"    {label:<14} {p:.2%}")
        print()

max_clean = max(scores["clean"])
min_toxic = min(scores["toxic"])
if min_toxic > max_clean:
    print(f"PASS: toxic texts score higher ({min_toxic:.2%}) than clean texts ({max_clean:.2%}).")
else:
    print(f"FAIL: a clean text scored {max_clean:.2%} but a toxic text only {min_toxic:.2%}.")

"""Fine-tune model_TEST.h5 so it handles short everyday messages better.

The original model was trained only on long Wikipedia comments. It flags
short messages like "I am gay" or "you are not stupid" as toxic and misses
direct threats. This script continues training on:

  * a sample of the Jigsaw training data (so the model keeps what it knew), and
  * template-generated short sentences (compliments, identity mentions,
    negations, slang, insults, threats, identity hate).

The words and templates below are chosen to be different from the ones in
test_cases.py where possible, so test_model.py measures generalisation.

Needs data/train.csv from the Jigsaw Toxic Comment Classification challenge.
Run with:  python finetune.py
"""
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app import MAX_LEN, MODEL_PATH, TOKENIZER_PATH
from preprocess import get_stop_words, processing

DATA_PATH = "data/train.csv"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42
CLEAN_SAMPLE = 40000   # clean Jigsaw comments used for fine-tuning
HOLDOUT = 10000        # Jigsaw comments kept aside to check for regressions
# How often each generated sentence is repeated. Clean sentences get more
# weight because false alarms on harmless messages are the main problem.
CLEAN_REPEAT = 6
TOXIC_REPEAT = 3
EPOCHS = 1

CLEAN = [0, 0, 0, 0, 0, 0]
INSULT = [1, 0, 0, 0, 1, 0]
THREAT = [1, 0, 0, 1, 0, 0]
HATE = [1, 0, 0, 0, 1, 1]

COMPLIMENT_WORDS = [
    "nice", "pretty", "adorable", "clever", "friendly", "helpful", "cool", "great",
    "fantastic", "polite", "honest", "brave", "strong", "patient", "thoughtful", "humble",
    "cheerful", "loyal", "gentle", "wise", "fun", "graceful", "supportive", "stunning",
    "elegant", "genius", "hilarious", "attractive", "gifted", "precious",
]
COMPLIMENT_TEMPLATES = [
    "you are {}", "you are very {}", "u are {}", "you're {}", "ur so {}", "she is {}",
    "he is so {}", "you seem {}", "you are the most {} person I know", "you look really {}",
]

IDENTITIES = [
    "gay", "lesbian", "muslim", "jewish", "christian", "black", "white", "asian",
    "mexican", "indian", "trans", "a woman", "an immigrant", "hindu", "african",
    "bisexual", "a refugee", "arab", "latino", "chinese", "catholic", "a feminist",
    "queer", "disabled", "atheist",
]
IDENTITY_TEMPLATES = [
    "I'm {}", "my brother is {}", "she is {} and proud", "he is {}", "as someone who is {}, I agree",
    "my teacher is {}", "our new manager is {}", "I'm {} and I love my family",
    "it is okay to be {}", "my mom is {}", "my dad is {}", "my wife is {}", "my husband is {}",
    "my roommate is {} and we get along great", "I am proud to be {}", "my coworker is {}",
    "is it hard being {}?", "I have been {} my whole life", "my uncle is {} and a doctor",
    "the author of this book is {}", "our family is {}", "my friends know I'm {}",
    "being {} is part of who I am", "my classmate is {} and very smart", "I'm {} and happy",
]

NEGATED_WORDS = [
    "silly", "boring", "weird", "a jerk", "rude", "lazy", "crazy", "a clown", "a moron",
    "fat", "a failure", "mean", "a liar", "gross", "a creep", "a coward", "trash",
    "retarded", "a bad person", "a waste of time", "an imbecile", "disgusting", "a joke",
    "a nobody", "garbage", "a dumbass", "a freak", "a parasite", "brainless", "a disgrace",
]
NEGATION_TEMPLATES = [
    "you are not {}", "you aren't {}", "I do not think you are {}", "no one said you are {}",
    "you are never {}", "he is not {}", "she's not {}", "I don't believe you're {}",
    "nobody believes you are {}", "nobody said you were {}", "no one thinks he is {}",
    "I never called you {}", "you were never {}", "nobody here thinks she is {}",
    "I would never say you are {}", "don't worry, you are not {}", "you are definitely not {}",
    "no way you are {}",
]

SLANG = [
    "that was lit", "this beat slaps", "I'm so hungry I could eat a horse", "she killed that song",
    "damn this is tasty", "what the heck", "hell no I'm not going", "the concert was mad fun",
    "you're insane at this game", "this homework is killing me", "I'm crazy about this show",
    "that trick was sick", "he destroyed the exam", "I hate waiting in line", "I hate traffic",
    "I hate broccoli", "that movie was dope", "you murdered that dance", "bloody hell it's cold",
    "this is ridiculous lol", "my feet are killing me", "the new update sucks", "I'm dead lol",
    "the test was brutal", "damn right", "you're a legend", "this cake is to die for",
    "I nearly died laughing", "what the hell happened here", "hell of a game",
    "I'm so stupid, I left my phone at home", "that was a dumb idea of mine",
    "he's a savage on the field", "you beast", "the band shredded", "go kill it at the interview",
    "shoot me a message later", "let's hit the gym", "break a leg tonight",
    "this puzzle is driving me nuts", "fight for your dreams", "we beat them 3 to 0",
    "I hate how good this is", "stop it, you're killing me", "you crazy genius",
    "hell yes", "what the hell is this song, it's great", "damn that was close",
    "damn it, I missed the bus", "oh hell, I forgot the milk", "the heat is killing me",
    "I hate rainy days", "I hate homework", "I hate cold coffee", "I hate being late",
    "I hate losing at cards", "I hate the dentist", "this phone sucks", "the wifi sucks today",
    "losing sucks", "that ride was sick", "sick moves bro", "your drawing is sick",
    "you killed that presentation", "we killed it tonight", "she killed it on stage",
    "I feel stupid for asking", "sorry, that was a dumb question", "my stupid alarm didn't ring",
    "this stupid printer jammed again", "that was a silly mistake of mine",
    "holy crap that's huge", "holy moly", "the exam was hell", "hell of a day",
    "the tickets cost a bomb", "crazy stuff happened today", "this is insane, I love it",
    "you're killing it", "he's a killer guitarist", "I'm dying to see that movie",
    "I could kill for a burger", "that joke killed me", "damn good coffee",
    "the pain is killing me", "stop, I'm dying here", "it's freezing as hell outside",
]

INSULT_WORDS = [
    "a moron", "an imbecile", "a jerk", "a loser", "a stupid cow", "a dirty pig", "a fat idiot",
    "a clown", "garbage", "a failure", "a parasite", "a retard", "a disgrace", "an ugly freak",
    "a dumb idiot", "scum", "a stupid troll", "a nobody", "a worthless idiot", "an idiot",
]
INSULT_TEMPLATES = [
    "you are {}", "you're {}", "u r {}", "you are such {}", "you are really {}",
    "honestly you are {}", "shut up you are {}", "he is {}",
]

THREAT_VERBS = [
    "slaughter", "torture", "hang", "drown", "crush", "bury", "gut", "kill", "stab", "shoot",
    "beat", "hurt", "smash", "kick", "attack", "rape", "behead", "poison", "break", "cut",
]
THREAT_TEMPLATES = [
    "I'll {} you", "I will find you and {} you", "I'm gonna {} you", "I swear I will {} you",
    "I know where you live, I will {} you", "I will {} your whole family",
    "you better run before I {} you", "tomorrow I will {} you",
    "somebody needs to {} you", "someone ought to {} you", "someone is going to {} you soon",
    "I hope someone will {} you", "people like you should be taken out, I will {} you",
    "wait until I {} you", "I want to {} you", "next time I see you I will {} you",
]

HATED_GROUPS = [
    "gays", "muslims", "jews", "black people", "immigrants", "mexicans", "asians", "women",
    "trans people", "indians", "arabs", "refugees", "christians", "hindus", "lesbians",
    "africans", "latinos", "chinese people", "disabled people", "catholics",
]
HATE_TEMPLATES = [
    "{} are vermin", "kill all {}", "{} should be exterminated", "I hate all {}",
    "{} are subhuman trash", "{} do not deserve to live", "get rid of all {}",
    "{} are a disease",
]


def generate_augmented():
    clean = []
    clean += [t.format(w) for t in COMPLIMENT_TEMPLATES for w in COMPLIMENT_WORDS]
    clean += [t.format(w) for t in IDENTITY_TEMPLATES for w in IDENTITIES]
    clean += [t.format(w) for t in NEGATION_TEMPLATES for w in NEGATED_WORDS]
    clean += SLANG
    toxic = []
    toxic += [(t.format(w), INSULT) for t in INSULT_TEMPLATES for w in INSULT_WORDS]
    toxic += [(t.format(w), THREAT) for t in THREAT_TEMPLATES for w in THREAT_VERBS]
    toxic += [(t.format(w), HATE) for t in HATE_TEMPLATES for w in HATED_GROUPS]

    rows = [(s, CLEAN) for s in clean] * CLEAN_REPEAT + toxic * TOXIC_REPEAT
    print(f"Generated {len(clean)} clean and {len(toxic)} toxic short sentences.")
    return pd.DataFrame(rows, columns=["comment_text", "labels"])


def check_test_overlap(aug, stop_words):
    """Report how many test cases the model will have seen during fine-tuning."""
    from test_cases import CASES
    train_raw = {t.lower() for t in aug["comment_text"]}
    train_clean = {processing(t, stop_words) for t in aug["comment_text"]}
    same_text = sum(text.lower() in train_raw for text, _, _ in CASES)
    same_input = sum(processing(text, stop_words) in train_clean for text, _, _ in CASES)
    print(f"Test cases copied into training data: {same_text}/{len(CASES)}")
    print(f"Test cases whose cleaned text matches a training sentence: {same_input}/{len(CASES)}")


def encode(texts, tokenizer):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")


def mean_auc(y_true, y_pred):
    return np.mean([roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(len(LABEL_COLS))])


def main():
    random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    stop_words = get_stop_words()

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)

    # Split Jigsaw into a holdout set and a fine-tuning sample.
    df = pd.read_csv(DATA_PATH).sample(frac=1, random_state=SEED)
    holdout, rest = df.iloc[:HOLDOUT], df.iloc[HOLDOUT:]
    toxic_rows = rest[rest[LABEL_COLS].sum(axis=1) > 0]
    clean_rows = rest[rest[LABEL_COLS].sum(axis=1) == 0].iloc[:CLEAN_SAMPLE]
    jigsaw = pd.concat([toxic_rows, clean_rows])

    print(f"Preprocessing {len(jigsaw) + len(holdout)} Jigsaw comments...")
    x_jig = encode([processing(t, stop_words) for t in jigsaw["comment_text"]], tokenizer)
    y_jig = jigsaw[LABEL_COLS].values
    x_hold = encode([processing(t, stop_words) for t in holdout["comment_text"]], tokenizer)
    y_hold = holdout[LABEL_COLS].values

    aug = generate_augmented()
    check_test_overlap(aug, stop_words)
    x_aug = encode([processing(t, stop_words) for t in aug["comment_text"]], tokenizer)
    y_aug = np.array(aug["labels"].tolist())

    x_train = np.concatenate([x_jig, x_aug])
    y_train = np.concatenate([y_jig, y_aug]).astype("float32")
    order = np.random.permutation(len(x_train))
    x_train, y_train = x_train[order], y_train[order]

    auc_before = mean_auc(y_hold, model.predict(x_hold, batch_size=512, verbose=0))
    print(f"Holdout mean AUC before fine-tuning: {auc_before:.4f}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128,
              validation_data=(x_hold, y_hold.astype("float32")), verbose=2)

    auc_after = mean_auc(y_hold, model.predict(x_hold, batch_size=512, verbose=0))
    print(f"Holdout mean AUC after fine-tuning:  {auc_after:.4f}")

    model.save(MODEL_PATH)
    print(f"Saved fine-tuned model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

# VirtuShield - Hate Speech Classifier

A simple multi-label text classifier that scores a comment on six categories:
toxic, severe toxic, obscene, threat, insult and identity hate.

The model is a Bidirectional LSTM trained on the
[Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset,
served through a Streamlit web app.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app for classifying text |
| `model_TEST.h5` | Trained LSTM model (stored with Git LFS) |
| `tokenizer.pkl` | Keras tokenizer fitted on the training text |
| `hateClassifier_EDA.ipynb` | Exploratory data analysis |
| `hateClassifier_LSTM.ipynb` | Preprocessing and model training |

## Run locally

Requires Python 3.9 - 3.11 (TensorFlow 2.15 does not support newer versions) and [Git LFS](https://git-lfs.com).

```bash
git lfs install
pip install -r requirements.txt
streamlit run app.py
```

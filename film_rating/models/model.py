import os
from catboost.text_processing import Tokenizer
import nltk
from nltk.corpus import stopwords, wordnet
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import pandas as pd

MODEL_DIR = os.path.join(os.getcwd(), "film_rating/models")

cb_tokenizer = Tokenizer(lowercasing=True, separator_type="BySense", token_types=["Word", "Number"])
stop_words = stopwords.words("english")
lemmatizer = nltk.stem.WordNetLemmatizer()
hf_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb", clean_up_tokenization_spaces=True)

checkpoint = os.path.join(MODEL_DIR, "hf_clf")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.eval()

with open(os.path.join(MODEL_DIR, "cb_clf.pkl"), "rb") as f:
	clf = pickle.load(f)
with open(os.path.join(MODEL_DIR, "cb_reg.pkl"), "rb") as f:
	reg = pickle.load(f)

def predict(text):
    text = " ".join([lemmatizer.lemmatize(token) for token in cb_tokenizer.tokenize(text) if token not in stop_words])
    inputs = hf_tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    logits = model(inputs["input_ids"], inputs["attention_mask"]).logits[:,0].detach().numpy()
    df = pd.DataFrame({"text": text, "logits": logits}).assign(proba=lambda x: clf.predict_proba(x)[:,0])
    return clf.predict(df)[0], round(reg.predict(df)[0], 0).astype("int")

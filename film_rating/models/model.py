import os
import pickle
import numpy as np

""" Uncomment for Bert prediction."""
"""
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
import datasets
from datasets import Dataset
import transformers
from transformers import BertTokenizer, AutoConfig, AutoModel, modeling_outputs
import torch
import torch.nn as nn"""


MODEL_DIR = os.path.join(os.getcwd(), 'film_rating/models')



def predict(text):
	with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
		vectorizer = pickle.load(f)
	with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
		le = pickle.load(f)
	with open(os.path.join(MODEL_DIR, "clf.pkl"), "rb") as f:
		clf = pickle.load(f)
	with open(os.path.join(MODEL_DIR, "reg.pkl"), "rb") as f:
		reg = pickle.load(f)
		
	X = vectorizer.transform([text])
	label = le.inverse_transform(clf.predict(X))[0]
	rating = np.around(reg.predict(X)).astype(int)[0]
	return "Label: {}; Rating: {} ".format(label, rating)


"""
class ImdbDataset():
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def _lemmatizer(self, doc):
        doc = " ".join(str(doc).split())
        doc = [token.lemma_ for token in nlp(doc) if token.lemma_ not in stop_words]
        return " ".join(doc)

    def _lemmatize(self, text):
        text = text.apply(self._lemmatizer)
        return text

    def _label_encoder(self, labels):
        return [1 if label == "pos" else 0 for label in labels]

    def _preprocess_function(self, batch, tokenizer, max_length):
        return tokenizer(batch['text'], truncation=True, max_length=max_length)

    def from_df(self, data):
        data = Dataset.from_dict({
            "text": self._lemmatize(data["text"]), 
            "labels": self._label_encoder(data["label"]),
            "rating": data["rating"]}
        )
        tokenized_data = data.map(
            self._preprocess_function, 
            batched=True, 
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": 512}
        )
        tokenized_data = tokenized_data.class_encode_column("labels")
        return tokenized_data
        
        
        
class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels, hid_dim=1024):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_channels),
        )

    def forward(self, x):
        return self.fc(x)



class ImdbBertClassifier(nn.Module):
    def __init__(self, out_features):
        super(ImdbBertClassifier, self).__init__()
        self.out_features = out_features
        self.config = AutoConfig.from_pretrained(checkpoint, output_attentions=True, attn_implementation="eager")
        self.backbone = AutoModel.from_pretrained(checkpoint, config=self.config)
        in_features = self.backbone.pooler.dense.out_features
        self.dropout = nn.Dropout(0.1)
        self.classifier = Classifier(in_features, out_features)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        out = self.dropout(out.pooler_output)
        logits = self.classifier(out)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        out = modeling_outputs.TokenClassifierOutput({"loss": loss, "logits": logits})
        return out



def predict_label(path, input_ids, attention_mask):
    model = torch.load(path, weights_only=False)
    model.eval()
    model.cpu()
    preds = model(input_ids, attention_mask)[0].detach().numpy()
    label = "pos" if np.argmax(preds, axis=1) else "neg"
    return label


def predict_rating(path, input_ids, attention_mask, rating_ids=[1, 10, 2, 3, 4, 7, 8, 9]):
    model = torch.load(path, weights_only=False)
    model.eval()
    model.cpu()
    preds = model(input_ids, attention_mask)[0].detach().numpy()
    rating = rating_ids[np.argmax(preds, axis=1)[0]]
    return rating

def bert_predict(text):
    checkpoint = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(checkpoint, clean_up_tokenization_spaces=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = ImdbDataset(tokenizer)
    ds = dataset.from_df(pd.DataFrame({"text": [text], "label": [None], "rating": [None]}))
    batch = next(iter(ds))
    input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)
    label = predict_label(
        os.path.join(MODEL_DIR, "bert_clf2.pt"), input_ids, attention_mask
    )
    rating = predict_rating(
        os.path.join(MODEL_DIR, "bert_clf8.pt"), input_ids, attention_mask
    )
    out = "Label: {}; Rating: {} ".format(label, rating)
    return out
"""

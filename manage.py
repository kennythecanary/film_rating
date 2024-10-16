#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'film_rating.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
	""" Uncomment for Bert prediction."""

	""" import numpy as np
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
	import torch.nn as nn
	from film_rating.models.model import ImdbDataset, ImdbBertClassifier, Classifier """
	
	main()

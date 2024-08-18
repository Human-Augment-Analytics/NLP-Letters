from sklearn.model_selection import (
    train_test_split,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
)

from nltk import tokenize
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('maxent_treebank_pos_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scienceplots
import functools
import warnings
import re
import sys

dataset_path = "/Users/thomasorth/Downloads/results_s1_s2_roberta-base-finetuned-nlp-letters-s1_s2-pronouns-class-weighted.csv"
random_state = 100

# Load dataset
df = pd.read_csv(dataset_path, encoding='unicode_escape')
df = df[df["Prediction"] != df["GT"]]

gendered_words = ["they", "them", "their", "themself", "mx", "person", "person's"]

def masked_langauge(text: str):
    letter_words = text.split()
    words = [w for w in letter_words if w in gendered_words]
    return len(words)

# Preprocess
df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
df["count"] = df["Text"].apply(lambda letter: len(letter.split()))
df["masked_lang"] = df["Text"].apply(masked_langauge)
df["percent_gendered"] = df["masked_lang"] / df["count"]

print("Male count", df[df["GT"] == "male"]["percent_gendered"].mean())
print("Female count", df[df["GT"] == "female"]["percent_gendered"].mean())

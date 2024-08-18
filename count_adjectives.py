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

warnings.filterwarnings("ignore")

plt.style.use('ieee')

try:
    word_type = sys.argv[1]
except:
    word_type = "adj"

def keep_adj_adverbs(row: str) -> list:
    """Method to keep only adjectives and adverbs.

    Args:
        row: The row of text to process.
    
    Returns:
        Only adjectives and adverbs for a leter
    """
    words = row.split(" ")
    pos = nltk.pos_tag(words)
    if word_type == "adj":
        tags = ["JJ", "JJR", "JJS"]
    else:
        tags = ['RB', 'RBR', 'RBS', 'WRB']
    words = [word[0] for word in pos if word[1] in tags]
    return len(words)

dataset_path = "/Users/thomasorth/Downloads/results_TEXT_roberta-base-finetuned-nlp-letters-TEXT-pronouns-class-weighted.csv"
random_state = 100

# Load dataset
df = pd.read_csv(dataset_path, encoding='unicode_escape')
df = df[df["Prediction"] != df["GT"]]

# Preprocess
df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
df["count"] = df["Text"].apply(keep_adj_adverbs)
print(f"Word Type: {word_type}")
print("Male count", df[df["GT"] == "male"]["count"].mean())
print("Female count", df[df["GT"] == "female"]["count"].mean())

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

random_state = 100

# Load dataset
for data_type in ["TEXT", "s1_s2"]:
    for subset in ["TP", "FP", "FN", "TN"]:
        dataset_path = f"/Users/thomasorth/Downloads/results_{data_type}_roberta-base-finetuned-nlp-letters-{data_type}-all-class-weighted.csv"
        df = pd.read_csv(dataset_path, encoding='unicode_escape')
        df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
        if subset == "TP":
            df = df[(df["Prediction"] == 'female') & (df["GT"] == 'female')]
        if subset == "FP":
            df = df[(df["Prediction"] == 'male') & (df["GT"] == 'female')]
        if subset == "FN":
            df = df[(df["Prediction"] == 'female') & (df["GT"] == 'male')]
        if subset == "TN":
            df = df[(df["Prediction"] == 'male') & (df["GT"] == 'male')]

        gendered_words = ["they", "them", "their", "themself", "mx", "person", "person's"]

        def masked_langauge(text: str):
            letter_words = text.split()
            words = [w for w in letter_words if w in gendered_words]
            return len(words)

        # Preprocess
        df["count"] = df["Text"].apply(lambda letter: len(letter.split()))
        df["masked_lang"] = df["Text"].apply(masked_langauge)
        df['percent_gendered'] = df["masked_lang"] / df["count"]

        print(f"{data_type}, {subset}", f"Mean: {df['percent_gendered'].mean()}", f"Median: {df['percent_gendered'].median()}", f"Mode: {df['percent_gendered'].mode().mean()}")
        print()

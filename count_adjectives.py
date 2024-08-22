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

def keep_adj(row: str) -> list:
    """Method to keep only adjectives and adverbs.

    Args:
        row: The row of text to process.
    
    Returns:
        Only adjectives and adverbs for a leter
    """
    words = row.split(" ")
    pos = nltk.pos_tag(words)
    tags = ["JJ", "JJR", "JJS"]
    words = [word[0] for word in pos if word[1] in tags]
    return len(words)

def keep_adverbs(row: str) -> list:
    """Method to keep only adjectives and adverbs.

    Args:
        row: The row of text to process.
    
    Returns:
        Only adjectives and adverbs for a leter
    """
    words = row.split(" ")
    pos = nltk.pos_tag(words)
    tags = ['RB', 'RBR', 'RBS', 'WRB']
    words = [word[0] for word in pos if word[1] in tags]
    return len(words)

full_dataset = True
for data_type in ["TEXT", "s1_s2"]:
    if not full_dataset:
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
            
            df['count'] = df["Text"].apply(lambda letters: len(letters.split()))
            df['count'] = df["Text"].apply(keep_adverbs)
            df['adj_count'] = df["Text"].apply(keep_adj)
            print(f"{data_type}", f"{subset}", f"mean, median, mode word count: {df['count'].mean()}, {df['count'].median()}, {df['count'].mode().iloc[-1]}", f"mean, median, mode adj count: {df['adj_count'].mean()}, {df['adj_count'].median()}, {df['adj_count'].mode().iloc[-1]}", f"mean, median, mode  adv count: {df['adv_count'].mean()}, {df['adv_count'].median()}, {df['adv_count'].mode().iloc[-1]}")

    else:
        dataset_path = f"/Users/thomasorth/Downloads/sentence_sets.csv"
        df = pd.read_csv(dataset_path, encoding='unicode_escape')
        df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
        df['count'] = df[data_type].apply(lambda letters: len(letters.split()))
        df['adv_count'] = df[data_type].apply(keep_adverbs)
        df['adj_count'] = df[data_type].apply(keep_adj)

        male = df[df["APPLICANT_GENDER"] == "MALE"]
        female = df[df["APPLICANT_GENDER"] == "FEMALE"]
        print(f"{data_type}", f"mean, median for word count male: {male['count'].mean()}, {male['count'].median()}", f"mean, median for word count female: {female['count'].mean()}, {female['count'].median()}")
        print(f"{data_type}", f"mean, median for adv count male: {male['adv_count'].mean()}, {male['adv_count'].median()}", f"mean, median for word count female: {female['adv_count'].mean()}, {female['adv_count'].median()}")
        print(f"{data_type}", f"mean, median for word count male: {male['adj_count'].mean()}, {male['adj_count'].median()}", f"mean, median for word count female: {female['adj_count'].mean()}, {female['adj_count'].median()}")
        print()

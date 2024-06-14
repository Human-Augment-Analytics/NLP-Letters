from sklearn.model_selection import (
    train_test_split,
    validation_curve, 
    ValidationCurveDisplay,
    StratifiedKFold,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from nltk import tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download("names")

from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scienceplots
import functools
import warnings

warnings.filterwarnings("ignore")

plt.style.use('ieee')

s1_s2 = False
model = "rf"

text_column = "full_text" if not s1_s2 else "s1_s2"

def remove_stopwords(remove: int) -> list:
    """Method to remove stopwords and other items from text,

    Args:
        remove: Remove that step from the process.
        text: The text from the pandas row to process.
    
    Returns:
        The processed text.
    """
    removal = []
    if remove == 0 or remove == 1:
        stopwords = nltk.corpus.stopwords.words('english')
        if remove == 1:
            pronouns = ["he", "him", "his", "himself", "she", "her", "hers", "herself"]
            for p in pronouns:
                stopwords.remove(p)
        removal.extend(stopwords)
    
    if remove == 2:
        names = nltk.corpus.names
        female_names = list(map(lambda name: name.lower(), names.words('female.txt')))
        male_names = list(map(lambda name: name.lower(), names.words('male.txt')))
        removal.extend(female_names)
        removal.extend(male_names)
    if remove == 3:
        removal.extend(["mr", "ms", "mrs", "he", "him", "his", "she", "her", "hers"])
    if remove == 4:
        removal.extend(["first_name", "last_name", "middle_name"])
    return removal

dataset_path = "sentence_sets_trimmed.csv"
random_state = 100

# Load dataset
df = pd.read_csv(dataset_path, encoding='unicode_escape')

# Preprocess
for i in [0,1,2,3,4,5]:
    df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
    remove_ablation = remove_stopwords(i)

    train, test = train_test_split(df, test_size=0.2, random_state=random_state)
    train_x = train[text_column]
    train_y = train["applicant_gender"]
    train_y[train_y == "female"] = 0.0
    train_y[train_y == "male"] = 1.0
    train_y = train_y.values.astype(np.float32)


    test_x = test[text_column]
    test_y = test["applicant_gender"]
    test_y[test_y == "female"] = 0.0
    test_y[test_y == "male"] = 1.0
    test_y = test_y.values.astype(np.float32)

    count_vectorizer = CountVectorizer(stop_words=remove_ablation)
    train_x_processed = count_vectorizer.fit_transform(train_x)
    test_x_processed = count_vectorizer.transform(test_x)
    clf = None
    if model == "svm":
        best_parameters = {
            "kernel": "rbf",
            "C": 1,
            "class_weight": "balanced",
        }
        clf = SVC(**best_parameters)
    if model == "rf":
        best_parameters = {
            "n_estimators": 10,
            "ccp_alpha": 0.001,
            "class_weight": "balanced",
        }
        clf = RandomForestClassifier(**best_parameters)
    if model == "xgb":
        clf = xgb.XGBClassifier(scale_pos_weight=1)
    clf.fit(
        train_x_processed, train_y
    )
    predicted = clf.predict(test_x_processed)
    f1 = f1_score(predicted, test_y, average="macro")
    accuracy = accuracy_score(predicted, test_y)
    print(f"model: {model}, i: {i}, Macro F1: {f1}, Accuracy: {accuracy}")
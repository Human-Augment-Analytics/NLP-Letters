from sklearn.model_selection import (
    train_test_split,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    f1_score,
    accuracy_score,
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
model = "svm"

text_column = "full_text" if not s1_s2 else "s1_s2"

def remove_stopwords(remove: int) -> list:
    """Method to remove stopwords and other items from text,

    Args:
        remove: Remove that step from the process.
    
    Returns:
        The set of words to remove.
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
for i in [5]:
    df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
    remove_words = []
    train, test = train_test_split(df, test_size=0.2, random_state=random_state)
    train_x = train[text_column]
    train_y = train["applicant_gender"]


    test_x = test[text_column]
    test_y = test["applicant_gender"]

    count_vectorizer = CountVectorizer(stop_words=remove_words)
    train_x_processed = count_vectorizer.fit_transform(train_x)
    test_x_processed = count_vectorizer.transform(test_x)
    clf = None
    if model == "svm":
        best_parameters = {
            "kernel": "rbf",
            "C": 1,
            "class_weight": "balanced",
            "probability": True
        }
        clf = SVC(**best_parameters)
    if model == "rf":
        best_parameters = {
            "n_estimators": 10,
            "ccp_alpha": 0.001,
            "class_weight": "balanced",
        }
        clf = RandomForestClassifier(**best_parameters)
    clf.fit(
        train_x_processed, train_y
    )
    import eli5
    from eli5.lime import TextExplainer

    te = TextExplainer(random_state=random_state)
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(count_vectorizer, clf)
    te.fit(test[test["applicant_gender"] == "female"][text_column].iloc[0], pipe.predict_proba)
    with open("data.html", "w") as file:
        file.write(te.show_prediction(target_names=["female", "male"]).data)
    with open("weights.html", "w") as file:
        file.write(te.show_weights(target_names=["female", "male"]).data)

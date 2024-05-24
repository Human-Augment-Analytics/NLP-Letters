from sklearn.model_selection import train_test_split, validation_curve, ValidationCurveDisplay, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

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

plt.style.use('ieee')

def remove_stopwords(text: str) -> str:
    """Method to remove stopwords and other items from text,

    Args:
        text: The text from the pandas row to process.
    
    Returns:
        The processed text.
    """
    text = text.lower()
    stopwords = nltk.corpus.stopwords.words('english')
    names = nltk.corpus.names
    female_names = list(map(lambda name: name.lower(), names.words('female.txt')))
    male_names = list(map(lambda name: name.lower(), names.words('male.txt')))
    stopwords.extend(["first_name", "last_name", "middle_name", "mr", "ms", "mrs"])
    return " ".join([word for word in text.split(" ") if word not in stopwords])

dataset_path = "sentence_sets_trimmed.csv"
save_path = "results_rf"
random_state = 100

# Load dataset
Path(save_path).mkdir(exist_ok=True)
df = pd.read_csv(dataset_path, encoding='unicode_escape')

# Preprocess
df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
df["full_text"] = df["full_text"].apply(remove_stopwords)

train, test = train_test_split(df, test_size=0.2, random_state=random_state)
train_x = train["full_text"]
train_y = train["applicant_gender"]

count_vectorizer = CountVectorizer()
train_x_processed = count_vectorizer.fit_transform(train_x)

rf = RandomForestClassifier(ccp_alpha=0.001, class_weight="balanced")
cv = StratifiedKFold()
display = ValidationCurveDisplay.from_estimator(
    rf,
    train_x_processed,
    train_y,
    cv=cv,
    param_name="n_estimators",
    param_range=[10, 100, 200, 300],
    scoring="f1_macro",
    verbose=5,
    n_jobs=-1,
)
display.plot()
plt.title(f"Validation Curve for n_estimators parameter")
plt.savefig(Path(save_path) / f"rf_tune.png", bbox_inches="tight")
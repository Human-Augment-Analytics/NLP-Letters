from sklearn.model_selection import (
    train_test_split,
    validation_curve,
    ValidationCurveDisplay,
    StratifiedKFold
)
from sklearn.svm import SVC
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer
)
from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

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

s1_s2 = False

text_column = "full_text" if not s1_s2 else "s1_s2"

class CustomTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rows = []
        for row in X:
            data = [len(row), len(row.split()), len(nltk.sent_tokenize(row))]
            rows.append(data)
        return np.array(rows)


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
    stopwords.extend(female_names)
    stopwords.extend(male_names)
    stopwords = set(stopwords)
    return " ".join([word for word in text.split(" ") if word not in stopwords])

dataset_path = "sentence_sets_trimmed.csv"
save_path = "results_svm"
random_state = 100

# Load dataset
Path(save_path).mkdir(exist_ok=True)
df = pd.read_csv(dataset_path, encoding='unicode_escape')

# Preprocess
df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
df[text_column] = df[text_column].apply(remove_stopwords)

train, test = train_test_split(df, test_size=0.2, random_state=random_state)
train_x = train[text_column]
train_y = train["applicant_gender"]
train_x_processed = CustomTransformer().fit_transform(train_x)

kernel = "rbf"
svm = SVC(kernel=kernel, class_weight="balanced")
cv = StratifiedKFold()
display = ValidationCurveDisplay.from_estimator(
    svm,
    train_x_processed,
    train_y,
    cv=cv,
    param_name="C",
    param_range=[0.1, 1, 3, 5, 7, 10],
    scoring="f1_macro",
    verbose=5,
    n_jobs=-1,
)
display.plot()
plt.title(f"Validation Curve for C parameter with kernel {kernel}")
plt.savefig(
    Path(save_path) / f"svm_tune_{kernel}_{text_column}_{'tfidf' if tfidf else 'count'}.png",
    bbox_inches="tight"
)
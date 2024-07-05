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

warnings.filterwarnings("ignore")

plt.style.use('ieee')

s1_s2 = True
model = "rf"
text_column = "full_text" if not s1_s2 else "s1_s2"

def keep_adj_adverbs(row: str) -> list:
    """Method to keep only adjectives and adverbs.

    Args:
        row: The row of text to process.
    
    Returns:
        Only adjectives and adverbs for a leter
    """
    words = row.split(" ")
    pos = nltk.pos_tag(words)
    tags = ["RB", "RBR", "RBS", "JJ", "JJR", "JJS"]
    return " ".join([word[0] for word in pos if word[1] in tags])

dataset_path = "sentence_sets_trimmed.csv"
random_state = 100
save_path = "results_adj_adverbs"
Path(save_path).mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(dataset_path, encoding='unicode_escape')

# Preprocess
df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
df[text_column] = df[text_column].apply(keep_adj_adverbs)

train, test = train_test_split(df, test_size=0.2, random_state=random_state)
train_x = train[text_column]
train_y = train["applicant_gender"]


test_x = test[text_column]
test_y = test["applicant_gender"]

count_vectorizer = CountVectorizer()
train_x_processed = count_vectorizer.fit_transform(train_x)
test_x_processed = count_vectorizer.transform(test_x)
clf = None
if model == "svm":
    best_parameters = {
        "kernel": "rbf",
        "C": 1,
        "class_weight": {"female": 8, "male": 1},
    }
    clf = SVC(**best_parameters)
if model == "rf":
    best_parameters = {
        "n_estimators": 10,
        "ccp_alpha": 0.001,
        "class_weight": {"female": 8, "male": 1},
    }
    clf = RandomForestClassifier(**best_parameters)
clf.fit(
    train_x_processed, train_y
)
predicted = clf.predict(test_x_processed)
f1 = f1_score(predicted, test_y, average="macro")
accuracy = accuracy_score(predicted, test_y)
print(f"model: {model}, s1_s2: {s1_s2}, Macro F1: {f1}, Accuracy: {accuracy}")
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_confusion_matrix(cm, class_names):

	fig, ax = plt.subplots()

	fig, ax = plt.subplots()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	im = ax.imshow(cm)
	fig.colorbar(im, cax=cax, orientation='vertical')

	ax.set_xticks(np.arange(len(class_names)))
	ax.set_yticks(np.arange(len(class_names)))

	ax.set_xticklabels(class_names)
	ax.set_yticklabels(class_names)

	ax.set_ylabel("True")
	ax.set_xlabel("Predicted")

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")

	for i in range(len(class_names)):
		for j in range(len(class_names)):
			text = ax.text(j, i, f"{np.around(cm[i, j], decimals=2)}%",
						ha="center", va="center", color="k")

	ax.set_title(f"Confusion Matrix for {model}, column: {text_column}")
	fig.tight_layout()
	plt.savefig(Path(save_path) / f"{model}_cm_{text_column}.png", bbox_inches="tight")
	plt.clf()

cm = confusion_matrix(test_y, predicted, labels=clf.classes_).astype(np.int32)
row_sums = cm.sum(axis=1)
cm = (np.around(cm / row_sums[:, np.newaxis], decimals=2) * 100).astype(np.int32)
plot_confusion_matrix(cm, clf.classes_)

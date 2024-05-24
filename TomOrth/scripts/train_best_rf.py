from sklearn.model_selection import train_test_split, validation_curve, ValidationCurveDisplay, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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

test_x = test["full_text"]
test_y = test["applicant_gender"]

count_vectorizer = CountVectorizer()
train_x_processed = count_vectorizer.fit_transform(train_x)
test_x_processed = count_vectorizer.transform(test_x)

best_parameters = {
    "n_estimators": 10,
    "ccp_alpha": 0.001,
    "class_weight": "balanced",
}
rf = RandomForestClassifier(**best_parameters)
rf.fit(
    train_x_processed, train_y
)

predicted = rf.predict(test_x_processed)
f1 = f1_score(predicted, test_y, average="macro")
accuracy = accuracy_score(predicted, test_y)
print(f"Macro F1: {f1}, Accuracy: {accuracy}")
cm = confusion_matrix(test_y, predicted, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf.classes_)
disp.plot()
plt.title("Confusion Matrix for RandomForest")
plt.savefig(Path(save_path) / f"rf_cm.png", bbox_inches="tight")
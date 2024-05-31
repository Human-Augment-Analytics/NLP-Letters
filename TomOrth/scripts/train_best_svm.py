from sklearn.model_selection import (
    train_test_split,
    validation_curve, 
    ValidationCurveDisplay,
    StratifiedKFold,
)
from sklearn.svm import SVC
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

plt.style.use('ieee')

s1_s2 = False

text_column = "full_text" if not s1_s2 else "s1_s2"

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

test_x = test[text_column]
test_y = test["applicant_gender"]

count_vectorizer = CountVectorizer()
train_x_processed = count_vectorizer.fit_transform(train_x)
test_x_processed = count_vectorizer.transform(test_x)

best_parameters = {
    "kernel": "rbf",
    "C": 1,
    "class_weight": "balanced",
}
svm = SVC(**best_parameters)
svm.fit(
    train_x_processed, train_y
)

predicted = svm.predict(test_x_processed)
f1 = f1_score(predicted, test_y, average="macro")
accuracy = accuracy_score(predicted, test_y)
print(f"Macro F1: {f1}, Accuracy: {accuracy}")

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

	ax.set_title("Confusion Matrix for SVM")
	fig.tight_layout()
	plt.savefig(Path(save_path) / f"svm_cm_{text_column}.png", bbox_inches="tight")
	plt.clf()

cm = confusion_matrix(test_y, predicted, labels=svm.classes_).astype(np.int32)
row_sums = cm.sum(axis=1)
cm = (np.around(cm / row_sums[:, np.newaxis], decimals=2) * 100).astype(np.int32)
plot_confusion_matrix(cm, svm.classes_)
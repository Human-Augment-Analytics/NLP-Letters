import pandas as pd
from datasets import Dataset

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
    pronouns = ["he", "him", "his", "himself", "she", "her", "hers", "herself"]
    for p in pronouns:
        stopwords.remove(p)
    """
    names = nltk.corpus.names
    female_names = list(map(lambda name: name.lower(), names.words('female.txt')))
    male_names = list(map(lambda name: name.lower(), names.words('male.txt')))
    stopwords.extend(["first_name", "last_name", "middle_name", "mr", "ms", "mrs"])
    stopwords.extend(female_names)
    stopwords.extend(male_names)
    stopwords = set(stopwords)
    """
    return " ".join([word for word in text.split(" ") if word not in stopwords])

dataset_path = "sentence_sets_trimmed.csv"
save_path = "results_setfit"
random_state = 100

# Load dataset
Path(save_path).mkdir(exist_ok=True)
df = pd.read_csv(dataset_path, encoding='unicode_escape')

# Preprocess
df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
#df[text_column] = df[text_column].apply(remove_stopwords)
train, test = train_test_split(df, test_size=0.2, random_state=random_state)
# convert to Dataset format
train = Dataset.from_pandas(train.head(100))
test = Dataset.from_pandas(test)

### Run training
from setfit import SetFitModel, SetFitTrainer, TrainingArguments
# Select a model
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
# training with Setfit
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

	ax.set_title("Confusion Matrix for SetFit")
	fig.tight_layout()
	plt.savefig(Path(save_path) / f"setfit_cm_{text_column}.png", bbox_inches="tight")
	plt.clf()

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(y_pred, y_test):
    y_pred = y_pred.astype(object)
    y_test = y_test.astype(object)
    y_pred[y_pred == 0] = "male"
    y_pred[y_pred == 1] = "female"

    y_test[y_test == 0] = "male"
    y_test[y_test == 1] = "female"

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["male", "female"]).astype(np.int32)
    row_sums = cm.sum(axis=1)
    cm = (np.around(cm / row_sums[:, np.newaxis], decimals=2) * 100).astype(np.int32)
    plot_confusion_matrix(cm, ["male", "female"])
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

trainer = SetFitTrainer(
    model=model,
    train_dataset=train, # to keep the code simple I do not create the df_train
    eval_dataset=test, # to keep the code simple I do not create the df_eval
    column_mapping={"full_text": "text", "applicant_gender": "label"},
    metric=compute_metrics,
)
trainer.train(args=TrainingArguments(num_epochs=(1,16), batch_size=(8,2)))
print(trainer.evaluate(test))
from sklearn.model_selection import train_test_split
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
from tqdm import tqdm

from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch

plt.style.use('ieee')

def plot_confusion_matrix(cm, class_names):

	# Colorbar based on: https://stackoverflow.com/a/39938019
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
			text = ax.text(j, i, f"{cm[i, j]*100}%",
						ha="center", va="center", color="k")

	ax.set_title("Confusion Matrix for Roberta")
	fig.tight_layout()
	plt.savefig("results_roberta/confusion_matrix.png", bbox_inches="tight")
	plt.clf()

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
save_path = "results_roberta"
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
train_y[train_y == "female"] = 0
train_y[train_y == "male"] = 1

test_x = test["full_text"]
test_y = test["applicant_gender"]
test_y[test_y == "female"] = 0
test_y[test_y == "male"] = 1

from torch import cuda
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import transformers
device = 'cuda' if cuda.is_available() else 'cpu'
import logging
logging.basicConfig(level=logging.ERROR)

MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizer.from_pretrained(
    'roberta-base',
    truncation=True,
    do_lower_case=True
)

class GenderData(Dataset):
    def __init__(self, text, gender, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.gender = gender
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.gender[index], dtype=torch.float)
        }

training_set = GenderData(train_x.values, train_y.values, tokenizer, MAX_LEN)
testing_set = GenderData(test_x.values, test_y.values, tokenizer, MAX_LEN)


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = RobertaClass()
model.to(device)

loss_function = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.7, 0.3]).to("cuda"))
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def train(epoch):
    print(f"EPOCH {epoch}")
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        if _%50==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training Loss per 50 steps: {loss_step}")
            print(f"Training Accuracy per 50 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return

for epoch in range(EPOCHS):
    train(epoch)

def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0; n_f1=0
    tot_cm = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            try:
                big_val, big_idx = torch.max(outputs.data, dim=1)
                nb_tr_steps += 1
                nb_tr_examples+=targets.size(0)
                n_correct += calcuate_accuracy(big_idx.detach().cpu().numpy(), targets.detach().cpu().numpy())
                n_f1 += f1_score(big_idx.detach().cpu().numpy(), targets.detach().cpu().numpy(), average="macro")
                pred = big_idx.detach().cpu().numpy().astype(object)
                targets_np = targets.detach().cpu().numpy().astype(object)
                targets_np[targets_np == 0] = "female"
                targets_np[targets_np == 1] = "male"
                pred[pred == 0] = "female"
                pred[pred == 1] = "male"
                cm = confusion_matrix(targets_np, pred, labels=["female", "male"])
                tot_cm += cm

                if _%5000==0:
                    accu_step = (n_correct*100)/nb_tr_examples
                    print(f"Validation Accuracy per 100 steps: {accu_step}")
            except Exception as e:
                print(e)
    epoch_accu = (n_correct*100)/nb_tr_examples
    epoch_f1 = (n_f1*100)/nb_tr_examples
    epoch_cm = tot_cm / nb_tr_examples
    row_sums = epoch_cm.sum(axis=1)
    epoch_cm = epoch_cm / row_sums[:, np.newaxis]
    epoch_cm = np.around(epoch_cm, decimals=2)
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu, epoch_f1, epoch_cm

acc, f1, cm = valid(model, testing_loader)
print(f"Accuracy and F1 on test data = {acc}, {f1}")
plot_confusion_matrix(cm, ["female", "male"])

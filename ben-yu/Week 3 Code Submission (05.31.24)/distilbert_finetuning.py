# -*- coding: utf-8 -*-
"""Distilbert Finetuning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14dxrnVPhFFJiEbyy4OhxsaY1j1reM7yu
"""

! pip install datasets transformers accelerate evaluate

from huggingface_hub import notebook_login

notebook_login()

!apt install git-lfs

from google.colab import drive
drive.mount('/content/drive')

from datasets import Dataset
import pandas as pd

#dataset_path = "/Users/benjamyu/workspace/NLP-Letters/data/sentence_sets_trimmed.csv"
dataset_path = "/content/drive/MyDrive/sentence_sets_trimmed.csv"
df = pd.read_csv(dataset_path, encoding='unicode_escape')
dataset = Dataset.from_pandas(df).rename_column("applicant_gender", "label").class_encode_column("label").train_test_split(test_size=0.2)

dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from datasets import load_metric
metric = load_metric('glue', 'mrpc')
#metric = evaluate.load('accuracy')

model_checkpoint = "distilbert-base-uncased"
batch_size = 16
task = "nlp-letters"
metric_name = "accuracy"
data_column = "full_text"
model_name = model_checkpoint.split("/")[-1]
num_labels = 2

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=True,
)

def preprocess_function(sample):
    return tokenizer(sample[data_column], truncation=True, padding=True)

accuracy = load_metric("accuracy")
precision = load_metric("precision")
recall = load_metric("recall")
f1 = load_metric("f1")

def compute_metrics(eval_pred):
    x, y = eval_pred
    preds = np.argmax(x, -1)
    return metric.compute(predictions=preds, references=y)


encoded_dataset = dataset.map(preprocess_function, batched=True)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)



trainer.train()

trainer.evaluate()

trainer.push_to_hub()

confusion_metric = evaluate.load("confusion_matrix")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return confusion_metric.compute(predictions=predictions, references=labels)

dataset['train'].features
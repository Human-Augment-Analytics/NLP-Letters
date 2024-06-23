from sklearn.model_selection import (
    train_test_split,
    validation_curve,
    ValidationCurveDisplay,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import os
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
save_path = "results_llm"
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

female_example = train[train["applicant_gender"] == "female"][text_column].iloc[0]
male_example = train[train["applicant_gender"] == "male"][text_column].iloc[0]

from openai import OpenAI


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

gpt_assistant_prompt = f"""
You are an experienced reviewer of letters of reccomendation who can discern the gender
of applicants, either female or male given the letter with no gendered information.
Here are two examples for reference.

female: {female_example}
male: {male_example}

When asked the applicant gender for a letter of reccomendation, simply reply with "female" or "male".
Those are your only options. You must choose one of them, even if you are unsure.
"""
results = []
for i in tqdm(range(30)):
    gpt_user_prompt = f"""What is the gender of the applicant this letter of reccomendation belongs to?

    {test[text_column].iloc[i]}
    """


    messages = [
        {"role": "system", "content": gpt_assistant_prompt},
        {"role": "user", "content": gpt_user_prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=4096,
        frequency_penalty=0.0
    )
    response_text = response.choices[0].message.content
    results.append(response_text)

print(f1_score(test["applicant_gender"].values[:30], results, average="macro")) # 22%
print(accuracy_score(test["applicant_gender"].values[:30], results)) # 73%


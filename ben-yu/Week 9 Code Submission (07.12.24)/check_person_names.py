import nltk
from nltk.corpus import names
import spacy
import pandas as pd
nlp = spacy.load('en_core_web_sm')
nltk.download("names")


dataset_path = "../../data/sentence_sets_trimmed.csv"

# Load dataset
df = pd.read_csv(dataset_path, encoding='unicode_escape')
nltk_names = []
female_names = list(map(lambda name: name.lower(), names.words('female.txt')))
male_names = list(map(lambda name: name.lower(), names.words('male.txt')))
nltk_names.extend(female_names)
nltk_names.extend(male_names)

found_persons = []
nltk_persons = []
for index, row in df.iterrows():
    doc = nlp(row['full_text'])
    persons = []
    nltk_hits = []
    for word in doc.ents:
        if word.label_ == 'PERSON' and ("FIRST_NAME" not in word.text  and "MIDDLE_NAME" not in word.text and "LAST_NAME" not in word.text):
            #print("Person found in row {}".format(index))
            #print(word.text, word.label_)
            persons.append(word.text)
        if word.text.lower() in nltk_names:
            nltk_hits.append(word.text)


    found_persons.append(persons)
    nltk_persons.append(nltk_hits)

df['found_persons'] = found_persons
df['nltk_persons'] = nltk_persons
df.to_csv('out.csv', index=False)



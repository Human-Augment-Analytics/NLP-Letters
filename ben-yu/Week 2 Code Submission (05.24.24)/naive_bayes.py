import pandas
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

sentence_data = pandas.read_csv('data/sentence_sets_trimmed.csv', encoding='unicode_escape', engine='python')


train = sentence_data.sample(frac=0.8,random_state=200)
test = sentence_data.drop(train.index)


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train.full_text)
X_train_tf = TfidfTransformer().fit_transform(X_train_counts)


text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

pipeline = text_clf.fit(train.full_text, train.applicant_gender)

print(np.mean(predicted == test.applicant_gender))

from collections import Counter
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

dataset_path = "data/charlotte_dataset_final.csv"
random_state = 100
save_path = "final_results"
label_column = "APPLICANT_GENDER"
text_column = "TEXT"
labels = ['FEMALE','MALE']


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_adjectives(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    # Apply part-of-speech tagging
    tagged_words = pos_tag(words)
    # Filter adjectives (JJ, JJR, JJS are POS tags for adjectives)
    adjectives = [word for word, pos in tagged_words if pos in ['JJ', 'JJR', 'JJS']]
    return adjectives

def get_adverbs(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    # Apply part-of-speech tagging
    tagged_words = pos_tag(words)
    # Filter adjectives (JJ, JJR, JJS are POS tags for adjectives)
    adverbs = [word for word, pos in tagged_words if pos in ['RB', 'RBR', 'RBS', 'WRB']]
    return adverbs


# Load dataset
df = pd.read_csv(dataset_path, encoding='unicode_escape')

female_letters = df.loc[df[label_column] == labels[0]]
female_adj_counts = Counter()
female_adv_counts = Counter()

male_letters = df.loc[df[label_column] == labels[1]]
male_adj_counts = Counter()
male_adv_counts = Counter()


for x in range(len(female_letters[text_column])):
    adjs = get_adverbs(female_letters[text_column].iloc[x])
    female_adv_counts.update(adjs)

print("Top 25 adverbs in female applicant text: {}".format(female_adv_counts.most_common(25)))

for x in range(len(male_letters[text_column])):
    adjs = get_adverbs(male_letters[text_column].iloc[x])
    male_adv_counts.update(adjs)

print("Top 25 adverbs in male applicant text: {}".format(male_adv_counts.most_common(25)))


#for x in range(len(female_letters[text_column])):
#    adjs = get_adjectives(female_letters[text_column].iloc[x])
#    female_adj_counts.update(adjs)
#
#print("Top 25 adjectives in female applicant text: {}".format(female_adj_counts.most_common(25)))
#
#for x in range(len(male_letters[text_column])):
#    adjs = get_adjectives(male_letters[text_column].iloc[x])
#    male_adj_counts.update(adjs)
#
#print("Top 25 adjectives in male applicant text: {}".format(male_adj_counts.most_common(25)))


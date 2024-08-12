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
    return adjectives, len(words)

def get_adverbs(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    # Apply part-of-speech tagging
    tagged_words = pos_tag(words)
    # Filter adjectives (JJ, JJR, JJS are POS tags for adjectives)
    adverbs = [word for word, pos in tagged_words if pos in ['RB', 'RBR', 'RBS', 'WRB']]
    return adverbs, len(words)


# Load dataset
df = pd.read_csv(dataset_path, encoding='unicode_escape')
total_adv_counts = Counter()

female_letters = df.loc[df[label_column] == labels[0]]
female_adj_counts = Counter()
female_adv_counts = Counter()

male_letters = df.loc[df[label_column] == labels[1]]
male_adj_counts = Counter()
male_adv_counts = Counter()


female_wc = 0
for x in range(len(female_letters[text_column])):
    adjs, wc = get_adverbs(female_letters[text_column].iloc[x])
    female_adv_counts.update(adjs)
    female_wc += wc

male_wc = 0
for x in range(len(male_letters[text_column])):
    adjs, wc= get_adverbs(male_letters[text_column].iloc[x])
    male_adv_counts.update(adjs)
    male_wc += wc

total_wc = 0
for x in range(len(df[text_column])):
    adjs, wc = get_adverbs(df[text_column].iloc[x])
    total_adv_counts.update(adjs)
    total_wc += wc

female_pcts = []
for w,c in female_adv_counts.items():
    if total_adv_counts[w] > 10:
        female_pcts.append((w, (female_adv_counts[w]/female_wc)/(total_adv_counts[w]/total_wc)))

female_pcts = sorted(female_pcts, key=lambda tup: tup[1],reverse=True)
print("Top Female adv by cond. prob: {}".format(female_pcts[:25]))

male_pcts = []
for w,c in male_adv_counts.items():
    if total_adv_counts[w] > 10:
        male_pcts.append((w, (male_adv_counts[w]/male_wc)/(total_adv_counts[w]/total_wc)))

male_pcts = sorted(male_pcts, key=lambda tup: tup[1],reverse=True)
print("Top Male adv by cond. prob: {}".format(male_pcts[:25]))

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


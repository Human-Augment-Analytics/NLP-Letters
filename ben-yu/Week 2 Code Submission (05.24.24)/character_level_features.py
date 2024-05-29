import seaborn as sns
import pandas

sentence_data = pandas.read_csv('data/sentence_sets_trimmed.csv', encoding='unicode_escape', engine='python')
sentence_data['character_count'] = sentence_data['TEXT'].str.len()
sentence_data['letter_count'] = sentence_data['TEXT'].str.count(r'[A-z]')
sentence_data['upper_case_count'] = sentence_data['TEXT'].str.count(r'[A-Z]')
sentence_data['white_space_count'] = sentence_data['TEXT'].str.count(' ')

sns.pairplot(data=sentence_data, hue="applicant_gender")


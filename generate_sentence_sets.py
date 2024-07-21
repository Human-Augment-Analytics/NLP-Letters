import pandas as pd

import spacy

from tqdm import tqdm

tqdm.pandas()

 

prons = {

        'FEMALE': ['she', 'hers', 'her', 'ms', 'miss', 'mrs'],

        'MALE': ['he', 'his', 'him', 'mr', 'mister'],

}

 

def get_sents(row):

        sents = {'S' + str(i+1):[] for i in range(3)}

 

        for sent in row['doc'].sents:

               sent = sent.text

               if any(x in sent for x in ['FIRST_NAME', 'MIDDLE_NAME', 'LAST_NAME']):

                       sents['S1'].append(sent)

               elif any(x in sent.lower() for x in prons[row['APPLICANT_GENDER']]):

                       sents['S2'].append(sent)

               else:

                       sents['S3'].append(sent)

        sents = {x:' * '.join(sents[x]) for x in sents}

        return pd.Series(sents)

 

 

nlp = spacy.load('en_core_web_lg')

 

data_path = 'data/charlotte_dataset_final.csv'

data = pd.read_csv(data_path, encoding='unicode_escape')

data = data.dropna(subset=['TEXT'])

data['TEXT'] = data['TEXT'].apply(lambda x: x.replace('|', 'I'))

data['doc'] = data['TEXT'].progress_apply(nlp)

data = data.join(pd.DataFrame(data.apply(get_sents, axis=1)))

data = data[[x for x in data.columns if x != 'doc']]

data["s1_s2"] = data["S1"] + " " + data["S2"]

data.to_csv('sentence_sets.csv', index=False)

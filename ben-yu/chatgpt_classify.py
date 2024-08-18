from enum import Enum
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
import time

client = OpenAI()

class Gender(str, Enum):
    female = "female"
    male = "male"

class ErrorGender(str, Enum):
    error = "error"

class ApplicantGender(BaseModel):
    gender: Gender

class Error(BaseModel):
    gender: ErrorGender

dataset_path = "data/results_TEXT_roberta-base-finetuned-nlp-letters-TEXT-pronouns-class-weighted.csv"
random_state = 100
save_path = "final_results"
label_column = "APPLICANT_GENDER"
text_column = "Text"
labels = ['female','male']

df = pd.read_csv(dataset_path, encoding='unicode_escape')
class_preds = []
gt_preds = []
text_preds = []
output_frame = []

missed_index = [200, 452, 703]

for index, row in df.iterrows():
    if index not in missed_index:
        continue
    text = row[text_column]
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "Determine if the inputted reference letter was written about a male or female applicant to a medical residency program."},
                {"role": "user", "content": text}
            ],
            response_format=ApplicantGender,
        )
        result = completion.choices[0].message.parsed
    except:
        result = Error(gender='error')
    output_frame.append((text, row['Prediction'], row['GT'], result.gender))
    #time.sleep(1)
    print("{} {} ".format(index, result))
#new_df = pd.DataFrame(output_frame, columns=["Text", "Prediction", "GT", "GPT Prediction"])
#new_df.to_csv(f"results_gpt.csv")

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score



df = pd.read_csv('results_gpt.csv', encoding='unicode_escape')
labels = ['female', 'male']

y = df['GPT Prediction']
preds = df['GT']

print(classification_report(y, preds, target_names=labels))
print("MCC: {}".format(matthews_corrcoef(y, preds)))
print("Balanced Accuracy: {}".format(balanced_accuracy_score(y, preds)))



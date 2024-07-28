# NLP-letters

This contains code for the NLP-letters project.

# Scripts

Check `ben-yu` and `TomOrth` folders have individual scripts. Check their respective readmes for information

# Running expermiments

For shallow models:
* python final_experiments.py -model svm -dataset full_text -preprocessing replace 
* python final_experiments.py -model svm -dataset s1_s2 -preprocessing replace 
* python final_experiments.py -model rf -dataset full_text -preprocessing replace 
* python final_experiments.py -model rf -dataset s1_s2 -preprocessing replace 

For BERT training, use `final_bert_training.py` or `Final_BERT_Training` notebook in Colab and comment and uncomment the sections of the script/notebook you need to run for the specific model.


# Conda Environment Setup

1. Install Anaconda https://docs.anaconda.com/free/anaconda/install/
2. Create conda env `conda env create -f environment.yml`
3. Activate environment `conda activate nlp-letters`

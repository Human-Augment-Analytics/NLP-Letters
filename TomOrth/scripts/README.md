# Scripts

These contain scripts written by Thomas Orth during summer 2024

These scripts assume you have a file called `sentence_sets_trimmed.csv`.

Ensure you have access to the data file to run. Rename if needed. Talk to Dr. Alexander for obtaining the data.

Early versions of the code used a column called `full_text` for the full, cleaned text. Later datasets changed this to `TEXT` at the end of the semester as we got new anonymized data. Update the conditional for `text_column` in order to work with these later files.

# Description

* tune_* scripts -> tuning of baseline shallow models
* train_* scripts -> training of shallow models and roberta
* shallow_* -> different experiments to further inspect shallow models
    * shallow_ablation -> ablation on preprocessing steps
    * shallow_adj_adverbs -> adjective and adverbs kept only for text
    * shallow_lime -> lime exploration
    * shallow_masking -> early text masking experiments
* setfit_runner -> Run SetFit on the text
* llm_run -> Run GPT-4 on the text
* eda.py -> early exploratory data analysis of the dataset

# Installation

For most scripts, install dependencies with `pip install -r main.requirements.txt`. For shallow_lime.py, you will need to do `pip install -r eli5.requirements.txt` as it requires a older version of sklearn.

For setfit_runner and train_roberta, these require google colab to run.

For llm_run, you will need to register an OpenAI account, get compute credits, and set the proper environment variable with your environment variable.
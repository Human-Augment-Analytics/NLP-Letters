# Running script

* Assumption: python 3.9 is installed on the user machine and the
command is `python`
* Download eda.py, requirements.txt and CSV file linked above to
the same folder
* Install dependencies by running: `python -m pip install -r require-
ments.txt –upgrade`
* Run: `python eda.py`

# Running expermiments

* python final_experiments.py -model svm -dataset full_text -preprocessing replace 
* python final_experiments.py -model svm -dataset s1_s2 -preprocessing replace 
* python final_experiments.py -model rf -dataset full_text -preprocessing replace 
* python final_experiments.py -model rf -dataset s1_s2 -preprocessing replace 


# Conda Environment Setup

1. Install Anaconda https://docs.anaconda.com/free/anaconda/install/
2. Create conda env `conda env create -f environment.yml`
3. Activate environment `conda activate nlp-letters`



# Flowchart
![flowchart](https://github.com/Human-Augment-Analytics/HAAG-Scripts-Repo/blob/main/Personal%20Folders/TomOrth/Week%201%20Code%20Submission%20(05.17.24)/eda.drawio.png)

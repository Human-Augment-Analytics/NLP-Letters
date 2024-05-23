from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import scienceplots

plt.style.use('ieee')
import pandas as pd
import numpy as np

from wordcloud import WordCloud, STOPWORDS

from nltk import tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download("names")


def remove_stopwords(text: str) -> str:
    """Method to remove stopwords and other items from text,

    Args:
        text: The text from the pandas row to process.
    
    Returns:
        The processed text.
    """
    text = text.lower()
    stopwords = nltk.corpus.stopwords.words('english')
    names = nltk.corpus.names
    female_names = list(map(lambda name: name.lower(), names.words('female.txt')))
    male_names = list(map(lambda name: name.lower(), names.words('male.txt')))
    stopwords.extend(["first_name", "last_name", "middle_name", "mr", "ms", "mrs"])
    return " ".join([word for word in text.split(" ") if word not in stopwords])

class FrequencyUtils:
    """Class for frequency metrics."""

    def __init__(
        self,
        male_df: pd.DataFrame,
        female_df: pd.DataFrame,
        column: str,
        save_path: Union[str, Path],
        name: str
    ):
        """Initializes an instance of FrequencyUtils.
        
        Args:
            male_df: The dataframe with male data.
            female_df: The dataframe with female data.
            column: The column to focus on for the plot.
            save_path: The location to save the plot.
            name: The name of the plot for saving.
        """
        self.female_df = female_df
        self.male_df = male_df
        self.column = column
        self.save_path = Path(save_path)
        self.name = name
    
    def execute(self) -> None:
        """Executes the plot generation."""

        self.female_df["mean_word_count"] = self.female_df[self.column].map(lambda letter: len(letter.split()))
        female_mean_word_count = self.female_df["mean_word_count"].mean()


        self.male_df["mean_word_count"] = self.male_df[self.column].map(lambda letter: len(letter.split()))
        male_mean_word_count = self.male_df["mean_word_count"].mean()

        labels = ["Female", "Male"]
        values = [female_mean_word_count, male_mean_word_count]
        plt.bar(labels, values,  label=labels, color=["red", "blue"])
        plt.title(f"EDA Metrics for {self.column}")
        plt.ylabel("Mean Word Count Per Letter")
        plt.savefig(str(self.save_path / f"{self.name}.png"), bbox_inches="tight")
        plt.clf()

class WordCloudUtils:
    """Class to generate a word cloud."""

    def __init__(
        self,
        df: pd.DataFrame,
        save_path: Union[str, Path],
        column: str,
        name: str
    ):
        """Initializes instance of WordCloudUtils.

        Args:
            male_df: The dataframe with male data.
            female_df: The dataframe with female data.
            column: The column to focus on for the plot.
            save_path: The location to save the plot.
            name: The name of the plot for saving.
        """
        self.df = df
        self.save_path = Path(save_path)
        self.column = column
        self.name = name

    def execute(self) -> None:
        """Executes word cloud generation."""
        stopwords = set(STOPWORDS)
        stopwords.update(["FIRST_NAME", "LAST_NAME", "MIDDLE_NAME", "mr", "ms", "mrs"])
        cloud = WordCloud(stopwords=stopwords).generate(' '.join(self.df[self.column]))
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(str(self.save_path / f"{self.name}.png"), bbox_inches="tight")
        plt.clf()

class ColumnDistributionUtils:
    """Utils for column distribution graphs."""

    def __init__(
        self,
        df: pd.DataFrame,
        column: str,
        save_path: Union[str, Path],
        name: str
    ):
        """Initializes an instance of ColumnDistributionUtils.

        Args:
            male_df: The dataframe with male data.
            female_df: The dataframe with female data.
            column: The column to focus on for the plot.
            save_path: The location to save the plot.
            name: The name of the plot for saving.
        """
        self.df = df
        self.column = column
        self.save_path = Path(save_path)
        self.name = name
    
    def execute(self):
        """Executes the plot creation."""
        data = self.df[self.column]
        _, _, autotexts = plt.pie(data.value_counts(), labels=data.unique().tolist(), autopct='%1.1f%%')
        for autotext in autotexts:
            autotext.set_color('white')
        plt.title(f"Distribution of {self.column} in dataset")
        plt.savefig(str(self.save_path / f"{self.name}.png"), bbox_inches="tight")
        plt.clf()


dataset_path = "sentence_sets_trimmed.csv"
save_path = "results"

# Load dataset
Path(save_path).mkdir(exist_ok=True)
df = pd.read_csv(dataset_path, encoding='unicode_escape')

# Preprocess
df.replace(to_replace=r'[^\w\s]', value='', regex=True, inplace=True)
df["full_text"] = df["full_text"].apply(remove_stopwords)

# Segment by gender
female_df = df[df["applicant_gender"] == "female"]
male_df = df[df["applicant_gender"] == "male"]

# Generate word clouds
WordCloudUtils(df, save_path, "full_text", "wordcloud").execute()
WordCloudUtils(female_df, save_path, "full_text", "wordcloud_female").execute()
WordCloudUtils(male_df, save_path, "full_text", "wordcloud_male").execute()

# Generate distribution pie charts
ColumnDistributionUtils(df, "applicant_gender", save_path, "gender_dist").execute()
ColumnDistributionUtils(df, "letter_type", save_path, "letter_typ_dist").execute()

# Generate frequency bar chart
FrequencyUtils(male_df, female_df, "full_text", save_path, "eda_chart").execute()

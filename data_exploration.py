import numpy as np
import pandas as pd
from tabulate import tabulate
from nltk import ngrams
import collections
import matplotlib.pyplot as plt


def validate_null_value(news_df):
    """
    Check the input dataframe for null values
    :param news_df: input dataframe
    :return: None
    """
    # check for a column with any null value
    print("List the count of null values for each column")
    sum_null_value = news_df.isnull().sum()
    print(sum_null_value)

    # check for a column with any null value
    print("List the count of nan values for each column")
    print(np.sum(pd.isna(news_df)))

    # validate if there is any null value
    is_null = news_df.isnull().values.any()
    print("Is there any null value? " + str(is_null))


def word_frequency(text: str):
    """
    Get frequency of individual words in a text corpus

    :param text: text corpus
    :return: Counter object of tuples of words and their frequencies
    """
    # get individual words
    tokenized = text.split()

    # count the frequency
    word_counter = collections.Counter(tokenized)

    return word_counter


def ngram_frequency(text: str, n: int = 2):
    """
    Get ngrams frequency in a text corpus

    :param text: the text of interest
    :param n: n for ngram. Default = 2
    :return: Counter object of tuples of ngrams and their frequencies
    """
    # get individual words
    tokenized = text.split()

    # get the list of all the ngrams
    ngram_list = ngrams(tokenized, n)

    # get the frequency of each ngram in the corpus
    ngram_counter = collections.Counter(ngram_list)

    return ngram_counter


def count_unique_words(sequences):
    """
    Count the number of unique words in a list of text sequences
    :param sequences: list of text sequences
    :return: number of unique words
    """
    joined_sequence = " ".join(sequences).lower()
    unique_words = list(word_frequency(joined_sequence).keys())
    return unique_words


def find_longest_sequence_length(sequences):
    """
    Find the length of the longest sequence
    :param sequences: list of text sequences
    :return: the length of the longest sequence
    """
    max_length = np.max([len(sequence) for sequence in sequences])
    return max_length


def visualize_composition(df):
    """
    Visualize the composition of the input dataset
    :param fake_news: fake news dataframe
    :param real_news: real news dataframe
    :return: None
    """
    # separate real from fake
    fake_news = df[df['is_fake'] == 1]
    real_news = df[df['is_fake'] == 0]

    # Categories of news
    # business      = 1
    # education     = 2
    # entertainment = 3
    # politics      = 4
    # sport         = 5
    # tech          = 6
    # celebrity     = 7
    labels = ['Business', 'Education', 'Entertainment', 'Politics', 'Sport', 'Technology', 'Celebrity']

    # Get the counts of the categories in reversed order
    fake_news_counts = fake_news['Category'].value_counts().tolist()
    real_news_counts = real_news['Category'].value_counts().tolist()
    # reverse the order
    fake_news_counts.reverse()
    real_news_counts.reverse()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, fake_news_counts, width, label='Fake')
    rects2 = ax.bar(x + width / 2, real_news_counts, width, label='Real')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Composition of the Fake News Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.xticks(rotation=90)
    plt.show()

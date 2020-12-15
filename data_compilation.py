import pandas as pd
import os
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clean_text(df, column, new_column):
    """
    Remove all punctuations and marks
    :param df: data frame to be cleaned
    :param text_field: name of column to be cleaned
    :param new_text_field: result column
    :return: the original data frame with the result column
    """
    df[new_column] = df[column].str.lower()
    df[new_column] = df[new_column].apply(
        lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", elem)
    )
    return df


def remove_stop_words(df, column: str):
    """
    Remove stop words from a column in a dataframe
    :param df: dataframe
    :param column: column of interest
    :return: the original dataframe with an additional column that has stopwords removed
    """
    df[column + " - no stopwords"] = df[column].apply(remove_stop_words_from_text)
    return df


def remove_stop_words_from_text(text):
    """
    Helper function to remove stopwords from a dataframe
    :param text: text to clean
    :return: text with no stopwords
    """
    # build a set of stop words
    stop_words = set(stopwords.words('english'))

    # tokenize the words in the text
    word_tokens = word_tokenize(text)

    # filer out stop words
    filtered_tokens = [w for w in word_tokens if not w in stop_words]

    # put the text together again
    filtered_text = " ".join(filtered_tokens)

    return filtered_text


def get_dataset_from_file(dataset: str, is_fake: bool, is_celeb: bool):
    """
    Return a data frame from files
    :param dataset: name of the dataset
    :param is_fake: True is the dataset is fake news else False
    :param is_celeb: True if the dataset is celebrity news else False
    :return: a data frame of the dataset
    """
    # list of news titles
    titles = []
    # list of news contents
    contents = []
    # category of the news
    categories = []

    quality = 'fake' if is_fake else 'legit'

    # grab data from celebrity dataset
    cwd = os.getcwd()
    dataset_dir = os.path.join(os.path.join(os.path.join(cwd, 'fakeNewsDatasets'), dataset), quality)

    for filename in os.listdir(dataset_dir):

        with open(os.path.join(dataset_dir, filename), 'r', encoding='UTF8') as file:
            # read and store the title
            title = file.readline()
            titles.append(title)

            # read and store the content
            content_lines = file.read().splitlines()  # remove new-line characters
            content = " ".join(content_lines)
            contents.append(content)

        # assign the category
        # business      = 1
        # education     = 2
        # entertainment = 3
        # politics      = 4
        # sport         = 5
        # tech          = 6
        # celebrity     = 7
        if not is_celeb:
            if filename[0:3] == "biz":
                categories.append(1)
            if filename[0:3] == "edu":
                categories.append(2)
            if filename[0:3] == "ent":
                categories.append(3)
            if filename[0:3] == "pol":
                categories.append(4)
            if filename[0:3] == "spo":
                categories.append(5)
            if filename[0:3] == "tec":
                categories.append(6)
        else:
            categories.append(7)

    newsDf = pd.DataFrame({'Title': titles, 'Content': contents, 'Category': categories})

    if is_fake:
        newsDf['is_fake'] = 1
    else:
        newsDf['is_fake'] = 0

    return newsDf


def build_real_news_dataframe(include_celeb: bool = False):
    """
    Build real news data frame
    :param include_celeb: whether or nor to include the celebrity dataset
    :return: the real news data frame
    """
    # real news dataset
    realDf = pd.DataFrame()

    # build the dataframe with no celebrity news
    realDf = realDf.append(get_dataset_from_file("fakeNewsDataset", False, False), ignore_index=True)

    # include data from the celebrity news
    if include_celeb:
        realDf = realDf.append(get_dataset_from_file("celebrityDataset", False, True), ignore_index=True)

    return realDf


def build_fake_news_dataframe(include_celeb: bool = False):
    """
    Build fake news data frame
    :param include_celeb: whether or not to include the celebrity dataset
    :return: the fake news data frame
    """
    # fake news dataset
    fakeDf = pd.DataFrame()

    # build the dataframe with no celebrity news
    fakeDf = fakeDf.append(get_dataset_from_file("fakeNewsDataset", True, False), ignore_index=True)

    # include data from the celebrity news
    if include_celeb:
        fakeDf = fakeDf.append(get_dataset_from_file("celebrityDataset", True, True), ignore_index=True)

    return fakeDf

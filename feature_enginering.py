import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim
import pandas as pd
import numpy as np
from tabulate import tabulate


stop_words = []

def process_feature_enginering(df):
    df_original = combine_two_columns(df, 'Title', 'Content', 'original')
    df_clean, stop_words = remove_stop_words(df_original)
    return df_clean, stop_words


def combine_two_columns(df, first_column, second_column, new_column):
    '''
    :param df: data fram
    :param first_column: first column to be combined
    :param second_column: seconed column to be combined
    :param new_column: the new column to be named
    :return: copy of the new data fram
    '''
    df_original = df.copy()
    df_original[new_column] = df_original[first_column].astype(str) + ' ' + df[second_column]

    return df_original

def remove_stop_words(df):
    '''
    :param df:
    :return:
    '''
    stop_words = stopwords.words('english')
    stop_words.extend(['lot', 'close', 'her', 'to', 'for', 'with', 'and'])
    df['clean'] = df['original'].apply(preprocess_stop_word)
    df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))
    print(tabulate(df.head(5), headers='keys', tablefmt='psql'))
    print(df.shape)
    '''
    list_of_words = []
    for i in df.clean:
        for j in i:
            list_of_words.append(j)
    total_words = len(list(set(list_of_words)))
    '''
    return df, stop_words


def preprocess_stop_word(text):
    '''
    :param text: text to be processed
    :param stop_words: list of stop words
    :return: return the processed text
    '''
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
        # if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)
    return result
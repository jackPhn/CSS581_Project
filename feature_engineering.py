import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim
import pandas as pd
import numpy as np
from tabulate import tabulate

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
import re

stop_words = []

def process_feature_engineering(df):
    df_original = combine_two_columns(df, 'Title', 'Content', 'original')
    df_clean, stop_words = remove_stop_words(df_original)
    total_words = find_total_words(df_clean)
    maxlen = find_max_token_length(df_clean)
    return df_clean, stop_words, total_words, maxlen


def combine_two_columns(df, first_column, second_column, new_column):
    '''
    :param df: dataframe
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
    :param df:dataframe
    :return:new datafream and stop_words
    '''
    stop_words = stopwords.words('english')
    stop_words.extend(['her', 'to', 'for', 'with', 'and'])
    df['clean'] = df['original'].apply(preprocess_stop_word)
    df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))
    print(tabulate(df.head(5), headers='keys', tablefmt='psql'))
    print(df.shape)
    return df, stop_words


# find the total list of words excluding stop words and redundent words
def find_total_words(df):
    '''
    :param df: dataframe
    :return: list of words
    '''
    list_of_words = []
    for i in df.clean:
        for j in i:
            list_of_words.append(j)
    total_words = len(list(set(list_of_words)))
    return total_words


def preprocess_stop_word(text):
    '''
    :param text: text to be processed
    :param stop_words: list of stop words
    :return: return the processed text
    '''
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
    return result


# find the max lengeth of a token
def find_max_token_length(df):
    '''
    :param df:
    :return:
    '''
    maxlen = -1
    for doc in df.clean_joined:
        tokens = nltk.wordpunct_tokenize(doc)
        if(maxlen < len(tokens)):
            maxlen = len(tokens)
    return  maxlen


def tfidf_transform(raw_data, tfidf_vectorizer=None):
    """
    Helper function to convert raw data of text into tf-idf matrix
    :param raw_data: raw text data
    :param tfidf_vectorizer: tfidf vectorizer from Scikit-Learn
    :return: tf-idf matrix and reference to the tf-idf vectorizer used
    """
    # tf-idf transformer
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(lowercase=True, smooth_idf=True)
        mat = tfidf_vectorizer.fit_transform(raw_data).todense()
    else:
        mat = tfidf_vectorizer.transform(raw_data).todense()

    return mat, tfidf_vectorizer


def vectorize_ngrams(raw_data, cv_ngram=None):
    """
    Helper function to convert raw data of text into matrix of ngram counts
    :param raw_data: raw text data
    :param cv_ngram: Scikit-Learn CountVectorizer
    :return: ngram count matrix and the CountVectorizer used
    """
    if cv_ngram is None:
        # count vectorizer
        # convert all words to lower case letters
        cv_ngram = CountVectorizer(analyzer='word', ngram_range=(3, 3), lowercase=True)
        # convert the input text data to a matrix of token counts
        mat = cv_ngram.fit_transform(raw_data).todense()
    else:
        mat = cv_ngram.transform(raw_data).todense()

    return mat, cv_ngram


def extract_features(X):
    """
    Extract features from news titles and contents
    :param df: two-column matrix of features (Title and Content)
    :return: feature matrix and feature extracting transformers
    """
    # Convert the titles to Tf-iDF matrix
    mat_title, tfidf_title = tfidf_transform(X[:, 0])

    # Convert the contents to Tf-iDF matrix
    mat_content, tfidf_content = tfidf_transform(X[:, 1])

    # count ngrams in the contents
    mat_ngram, cv_ngram = vectorize_ngrams(X[:, 1])

    X_mat = np.hstack((mat_title, mat_content))

    print("The size of the feature space is:", X_mat.shape)

    return {
        "cv_ngram": cv_ngram,
        "tfidf_content": tfidf_content,
        "tfidf_title": tfidf_title,
        "features": X_mat
    }


def tokenize_words(raw_data, max_length: int, tokenizer=None):
    """
    Tokenize words
    :param raw_data:    input list of texts
    :param max_length:  maximum length of an input sequence
    :param tokenizer:   a trained tokenizer. Create a new one if none
    :return:            list of tokenized input texts and trained tokenizer
    """
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    if tokenizer is None:
        tokenizer = Tokenizer(oov_token=oov_tok)
        tokenizer.fit_on_texts(raw_data)

    # pad the sequence
    sequences = tokenizer.texts_to_sequences(raw_data)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    return padded, tokenizer


def normalize(data):
    normalized = []
    for i in data:
        i = i.lower()
        # get rid of urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # get rid of non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized
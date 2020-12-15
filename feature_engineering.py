import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)


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
        cv_ngram = CountVectorizer(analyzer='word', ngram_range=(4, 4), lowercase=True)
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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tfdif_transform(raw_data, tfidf_vectorizer=None):
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


def ngram_vectorizer(raw_data, cv_ngram=None):
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


def word_embedding(raw_data, vocab_size, max_length, tokenizer=None):
    """

    :param raw_data:
    :param vocab_size:
    :param max_length:
    :param tokenizer:
    :return:
    """
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    if tokenizer is None:
        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(raw_data)

    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(raw_data)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    return padded, tokenizer


def decision_tree_model(df):
    """

    :param df: datafram of raw data
    :return:
    """
    # extract data
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    Y = Y.astype('int')

    # Convert the titles to Tf-iDF matrix
    mat_title, tfidf_title = tfdif_transform(X[:, 0])

    # Convert the contents to Tf-iDF matrix
    mat_content, tfidf_content = tfdif_transform(X[:, 1])

    # count ngrams in the contents
    mat_ngram, cv_ngram = ngram_vectorizer(X[:, 1])

    X_mat = np.hstack((mat_title, mat_content, mat_ngram))

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X_mat, Y, test_size=0.2, random_state=0, stratify=Y)

    # model
    model = GaussianNB()
    fit = model.fit(X_train, Y_train)

    # prediction
    Y_pred = fit.predict(X_test)
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
    print("Precision:", precision_score(Y_test, Y_pred))
    print("Recall:", recall_score(Y_test, Y_pred))
    print("F-Score:", f1_score(Y_test, Y_pred))

    return {
        "fit": fit,
        "cv_ngram": cv_ngram,
        "tfidf_content": tfidf_content,
        "tfidf_title": tfidf_title
    }


def make_prediction(pack, file_path):
    """

    :param pack:
    :param file_path:
    :return:
    """
    print("Make prediction for", file_path)
    fit = pack['fit']
    cv_ngram = pack['cv_ngram']
    tfidf_title = pack['tfidf_title']
    tfidf_content = pack['tfidf_content']

    title = []
    content = []

    # open file and get data
    with open(file_path) as file:
        # read and store the title
        title.append(file.readline())

        # read and store the content
        content_lines = file.readlines()
        content.append(" ".join(content_lines))

    # extract features
    mat_title, _ = tfdif_transform(raw_data=title, tfidf_vectorizer=tfidf_title)
    mat_content, _ = tfdif_transform(raw_data=content, tfidf_vectorizer=tfidf_content)
    mat_ngram, _ = ngram_vectorizer(raw_data=content, cv_ngram=cv_ngram)

    mat_sample = np.hstack((mat_title, mat_content, mat_ngram))

    # make prediction
    pred = fit.predict(mat_sample)

    if pred[0] == 1:
        print("This is fake news")
    else:
        print("This is legit news")


def basic_deep_learning_model(df):
    """

    :param df:
    :return:
    """
    vocab_size = 100
    embedding_dim = 16
    max_length = 100

    # extract data
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    Y = Y.astype('int')

    X_mat, tokenizer = word_embedding(raw_data=X[:, 1], vocab_size=vocab_size, max_length=max_length)
    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X_mat, Y, test_size=0.2, random_state=0, stratify=Y)

    # build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # train the model
    num_epoch = 10
    model.fit(X_train, Y_train, epochs=num_epoch, validation_data=(X_train, Y_train))

    return model

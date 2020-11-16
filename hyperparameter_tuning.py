import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from feature_engineering import (
    tfidf_transform,
    vectorize_ngrams,
    extract_features,
    tokenize_words
)
from tensorboard.plugins.hparams import api as hp  # for hyperparameter tuning


def none_dl_grid_search(df):
    """
    Using grid search to find the best model and hyperparameters
    :param df: raw data frame
    :return:
    """
    # extract data
    X = df[['Title', 'Content']].values
    Y = df['is_fake'].values
    labels = Y.astype('int')

    # extract the features from the data frame
    feature_pack = extract_features(X)
    features = feature_pack['features']

    # create models and parameters
    model_params = {
        'Logistic Regression': {
            'model': LogisticRegression(n_jobs=8),
            'params': {
                'C': [1, 5, 10]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [1, 5, 10]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(n_jobs=8),
            'params': {
                'n_estimators': [1, 5, 10]
            }
        }
    }

    # list of scores
    scores = []

    # iterate over the models
    for model_name, mp in model_params.items():
        print("Working on", model_name)
        clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        clf.fit(features, labels)
        scores.append({
            'model': model_name,
            'best_params': clf.best_params_,
            'best_score': clf.best_score_
        })

    # display the results
    resultsDF = pd.DataFrame(scores)
    resultsDF.columns = ['Model', 'Best Params', 'Best Score']
    print(scores)
    print(resultsDF)
    # save the results
    resultsDF.to_csv('output/none_dl_grid_search_results.csv')


# setup hyperparameter experiment
HP_NUM_UNITS = hp.HParam('num units', hp.Discrete([4, 8, 10, 16, 20, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'RMSprop']))
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('output/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')]
    )


def train_test_model(df, hparams):
    """
    Helper function for parameter tuning
    :param df:
    :param hparams:
    :return:
    """
    vocab_size = 19885 # max number of words possible in Tokenizer
    embedding_dim = 200
    max_length = 200

    # extract data
    X = df[['Title', 'Content']].values
    Y = df['is_fake'].values
    labels = Y.astype('int')

    # tokenize the words
    features, trained_tokenizer = tokenize_words(raw_data=X[:, 1], vocab_size=vocab_size, max_length=max_length)

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=0, stratify=Y)

    # neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation='relu'),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=hparams[HP_OPTIMIZER], loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=10)
    _, accuracy = model.evaluate(X_test, Y_test)
    return accuracy


def run(run_dir, hparams, df):
    """
    Helper function for hyperparameter tuning
    :param run_dir:
    :param hparams:
    :param df:
    :return:
    """
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(df, hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


def dl_grid_search(df):
    """
    Tune hyperparameters to select deep learning model
    :param df: input data frame containing raw data
    :return:
    """
    session_num = 0

    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                    HP_OPTIMIZER: optimizer,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('output/hparam_tuning/' + run_name, hparams, df)
                session_num += 1

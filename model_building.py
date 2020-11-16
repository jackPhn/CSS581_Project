import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, LSTM

from feature_engineering import (
    tfidf_transform,
    vectorize_ngrams,
    extract_features,
    tokenize_words
)

from keras_evaluation_metrics import (
    precision_m,
    recall_m,
    f1_m
)


def evaluate(fit, X_test, Y_test):
    """
    Evaluate a trained model for accuracy, precision, recall, f_score, and auc
    :param fit: trained model
    :param X_test: test features
    :param Y_test: test labels
    :return: a dictionary of metrics
    """
    pred = np.array(fit.predict(X_test))
    pred_proba = np.array(fit.predict_proba(X_test))

    return {
        "accuracy": accuracy_score(Y_test, pred),
        "precision": precision_score(Y_test, pred),
        "recall": recall_score(Y_test, pred),
        "f_score": f1_score(Y_test, pred),
        "auc": roc_auc_score(np.array(Y_test), pred_proba[:, 1])
    }


def cross_validate(model, features, labels):
    """
    Perform k fold cross validation
    :param model: untrained model
    :param features: feature matrix
    :param labels: label vector
    :return: a data frame of cross validation metrics
    """
    # Stratified 10-fold
    k = 10
    kfold = StratifiedKFold(n_splits=k, shuffle=True)

    # validation metrics
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f_score = 0.0
    auc = 0.0

    for train_indices, test_indices in kfold.split(features, labels):
        X_train = features[train_indices, :]
        Y_train = labels[train_indices]

        X_test = features[test_indices, :]
        Y_test = labels[test_indices]

        # train the model
        fit = model.fit(X_train, Y_train)

        # evaluate the model
        metrics = evaluate(fit, X_test, Y_test)

        accuracy += metrics['accuracy']
        precision += metrics['precision']
        recall += metrics['recall']
        f_score += metrics['f_score']
        auc += metrics['auc']

    # average and pack the results into a data frame
    values = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
        "Value": [accuracy / k, precision / k, recall / k, f_score / k, auc / k]
    }
    metrics_df = pd.DataFrame.from_dict(values)
    return metrics_df


def classical_models(df):
    """
    Build classical models and perform cross validation and evaluate the performance on test set
    :param df: raw data frame
    :return: a dictionary of the trained models and features extracting transformers
    """
    # extract data
    X = df[['Title', 'Content']].values
    Y = df['is_fake'].values
    labels = Y.astype('int')

    # extract the features from the data frame
    feature_pack = extract_features(X)
    features = feature_pack['features']

    # split the dataset 80 / 20 for train and test
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=0,
                                                        stratify=labels)

    # model
    models = {
        # "Logistic Regression": LogisticRegression(),
        # "Decision Tree": DecisionTreeClassifier(),
        # "Gaussian NB": GaussianNB(),
        # "Random Forest": RandomForestClassifier(),
        # "XGBoost": XGBClassifier(n_jobs=8),
        "SVM": SVC(gamma='auto', kernel='poly', probability=True),
    }

    # create a data frame to store validation metrics and test metrics
    metrics = {"Metrics": ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']}
    validation_metrics_df = pd.DataFrame.from_dict(metrics)
    test_metrics_df = pd.DataFrame.from_dict(metrics)

    # dictionary for storing weights
    trained_models = dict()

    for name, model in models.items():
        print("Working on", name)
        # k-fold cross validation
        metrics_df = cross_validate(model, X_train, Y_train)
        validation_metrics_df[name] = metrics_df["Value"]

        # train the model
        fit = model.fit(X_train, Y_train)
        trained_models[name] = fit

        # evaluate the model on the test set
        test_metrics = evaluate(fit, X_test, Y_test)
        # pack the results into a data frame
        values = {
            "Value": [test_metrics['accuracy'],
                      test_metrics['precision'],
                      test_metrics['recall'],
                      test_metrics['f_score'],
                      test_metrics['auc']]
        }
        test_metrics = pd.DataFrame.from_dict(values)
        test_metrics_df[name] = test_metrics["Value"]

    # display the results
    print("Cross validation results:")
    print(validation_metrics_df)
    print()

    print("Test set's evaluation results:")
    print(test_metrics_df)

    # Write the results to .csv files
    validation_metrics_df.to_csv('output/validation_results.csv')
    test_metrics_df.to_csv('output/test_results.csv')

    return {
        "models": trained_models,
        "cv_ngram": feature_pack['cv_ngram'],
        "tfidf_content": feature_pack['tfidf_content'],
        "tfidf_title": feature_pack['tfidf_title']
    }


def visualize_dl_training(history):
    """
    Draw plot of the training history for a deep learning model
    :param history: history object of the training
    :return: none
    """
    # visualize the training history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Test"], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def deep_learning_model(df):
    """
    Build a deep learning model
    :param df: input data frame containing raw data
    :return: trained neural network
    """
    vocab_size = 16876  # 3000
    embedding_dim = 100
    max_length = 1000  # 200

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
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    # LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPool1D(),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPool1D(),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.LSTM(20),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # GRU model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GRU(units=100, dropout=0.2, recurrent_dropout=0.2,
                            recurrent_activation='relu', activation='relu'),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    """

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision_m, recall_m, f1_m])
    model.summary()

    # train the model
    num_epoch = 20
    history = model.fit(X_train, Y_train, epochs=num_epoch, validation_data=(X_test, Y_test))

    # visualize the training history
    visualize_dl_training(history)

    # get the dictionary of words and frequencies in the corpus
    word_index = trained_tokenizer.word_index
    # reverse the key-value relationship in word_index
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    # get the weights for the embedding layer
    e = model.layers[0]
    weights = e.get_weights()[0]

    # write out the embedding vectors and metadata
    # To view the visualization, go to https://projector.tensorflow.org
    out_v = io.open('output/content_vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('output/content_meta.tsv', 'w', encoding='utf-8')
    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + '\n')
        out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
    # close files
    out_m.close()
    out_v.close()

    return {
        "fit": model,
        "tokenizer": trained_tokenizer,
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "max_length": max_length
    }


def make_prediction(model_pack, file_path: str, model_name: str):
    """
    Make prediction for a single file of news
    :param model_pack: contained model weights and feature extracting transformers
    :param file_path: full file system path to the .txt file containing the news
    :param model_name: name of the model to use
    :return: none
    """
    print("Make prediction for", file_path)

    title = []
    content = []

    # open file and get data
    with open(file_path) as file:
        # read and store the title
        title.append(file.readline())

        # read and store the content
        content_lines = file.read().splitlines()
        content.append(" ".join(content_lines))

    if model_name == "dl":
        # deep learning model
        fit = model_pack['fit']
        trained_tokenizer = model_pack['tokenizer']
        vocab_size = model_pack['vocab_size']
        max_length = model_pack['max_length']
        sample, _ = tokenize_words(content, vocab_size, max_length, trained_tokenizer)
    else:
        # other classical models
        models = model_pack['models']
        fit = models[model_name]
        cv_ngram = model_pack['cv_ngram']
        tfidf_title = model_pack['tfidf_title']
        tfidf_content = model_pack['tfidf_content']

        # extract features
        mat_title, _ = tfidf_transform(raw_data=title, tfidf_vectorizer=tfidf_title)
        mat_content, _ = tfidf_transform(raw_data=content, tfidf_vectorizer=tfidf_content)
        mat_ngram, _ = vectorize_ngrams(raw_data=content, cv_ngram=cv_ngram)
        sample = np.hstack((mat_title, mat_content, mat_ngram))

    # make prediction
    pred = fit.predict(sample)

    # Cutoff threshold is 0.5
    if pred[0] > 0.5:
        print("This is fake news")
    else:
        print("This is legit news")


def create_pad_sequence(df, total_words, maxlen):
    x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.is_fake, test_size=0.2)
    tokenizer = Tokenizer(new_words=total_words)
    # update internal vocabulary based on a list of tests
    tokenizer.fit_on_texts(x_train)
    # transformation each text into a sequences integer
    train_sequences = tokenizer.texts_to_sequences(x_train)
    test_sequences = tokenizer.texts_to_sequences(x_train)
    pad_train = pad_sequences(train_sequences, maxlen=maxlen, padding='post', truncating='post')
    pad_test = pad_sequences(test_sequences, maxlen=maxlen, padding='post', truncating='post')


def build_ltsm_model(padded_train, total_words, y_train):
    # create sequential model
    model = Sequential()

    # embedding layer
    model.add(Embedding(total_words, output_dim=128))

    # Bi-directional RNN/LSTM
    model.add(Bidirectional(LSTM(128)))

    # Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    y_train = np.asarray(y_train)

    # train the model
    print(model.fit(padded_train, y_train, batch_size=64, validation_split=0.1, epochs=2))

    return model


def predict_stml_model(model, padded_test, y_test):
    pred = model.predict(padded_test)
    prediction = []
    # if the predicted value is > 0.5 it is real else it is fake
    for i in range(len(pred)):
        if pred[i] > 0.5:
            prediction.append(1)
        else:
            prediction.append(0)

    # getting the meassurement
    accuracy = accuracy_score(list(y_test), prediction)
    precision = precision_score(list(y_test), prediction)
    recall = recall_score(list(y_test), prediction)
    f1score = f1_score(list(y_test), prediction)
    auc = roc_auc_score(list(y_test), prediction)

    print("STML Model Accuracy: ", accuracy)
    print("STML Model Precision: ", precision)
    print("STML Model Recall: ", recall)
    print("STML Model F1_score: ", f1score)
    print("STML Model AUC: ", auc)

    return prediction

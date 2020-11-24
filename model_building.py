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
    roc_auc_score
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import tensorflow as tf
from keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, LSTM, Dropout, GRU
from keras.optimizers import Adam

from feature_engineering import (
    tfidf_transform,
    vectorize_ngrams,
    extract_features,
    tokenize_words,
    process_feature_engineering,
    normalize
)

from data_visualization import(
    visualize_confusion_matrix
)

from keras_evaluation_metrics import (
    precision_m,
    recall_m,
    f1_m
)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def evaluate(fit, X_test, Y_test, is_dl: bool = False):
    """
    Evaluate a trained model for accuracy, precision, recall, f_score, and auc
    :param fit: trained model
    :param X_test: test features
    :param Y_test: test labels
    :param is_dl: whether the input model is deep learning
    :return: a dictionary of metrics
    """
    predictions = np.array(fit.predict(X_test))
    #predictions_proba = np.array([])

    # deal with deep learning model
    pred = []
    if is_dl:
        for i in range(len(predictions)):
            # cutoff threshold is 0.5
            if predictions[i] > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        predictions_proba = predictions
        predictions = np.array(pred)
    else:
        # deal with classical model
        predictions_proba = np.array(fit.predict_proba(X_test))
        # select the predictions for positive label
        predictions_proba = predictions_proba[:, 1]

    return {
        "accuracy": accuracy_score(Y_test, predictions),
        "precision": precision_score(Y_test, predictions),
        "recall": recall_score(Y_test, predictions),
        "f_score": f1_score(Y_test, predictions),
        "auc": roc_auc_score(np.array(Y_test), predictions_proba)
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
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=0, stratify=labels)

    # model
    models = {
        "Logistic Regression": LogisticRegression(),
        "Gaussian NB": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(n_jobs=8),
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
    plt.figure()
    # visualize the training history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Test"], loc='upper left')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def visualize_trained_word_embedding(model, trained_tokenizer, vocab_size):
    """
    Save files for viewing with Tensorflow Projector
    Go to https://projector.tensorflow.org and load the .tsv file
    :param model: the trained model
    :param trained_tokenizer: trained word tokenizer
    :param vocab_size: size of the vocabulary
    :return: None
    """
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


def build_nn_model(vocab_size, embedding_dim, max_length):
    """
    Construct a neural network with word embedding
    :param vocab_size: size of the vocabulary
    :param embedding_dim: embedding dimension
    :param max_length: max length of the input sequences
    :return: a keras model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def build_lstm_model2(vocab_size, embedding_dim, max_length):
    """
    Construct a LSTM model with word embedding
    :param vocab_size: size of the vocabulary
    :param embedding_dim: embedding dimension
    :param max_length: max length of the input sequences
    :return: a keras model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=4),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.LSTM(10),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def build_gru_model(vocab_size, embedding_dim, max_length):
    """
    Construct a GRU model with word embedding
    :param vocab_size: size of the vocabulary
    :param embedding_dim: embedding dimension
    :param max_length: max length of the input sequences
    :return: a keras model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        tf.keras.layers.GRU(units=64, dropout=0.2, recurrent_dropout=0.2,
                            recurrent_activation='sigmoid', activation='tanh'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def build_bidirectional_lstm_model(vocab_size, embedding_dim, max_length):
    """
    Construct a bidirectional LSTM model with word embedding
    :param vocab_size: size of the vocabulary
    :param embedding_dim: embedding dimension
    :param max_length: max length of the input sequences
    :return: a keras model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def build_combined_cnn_lstm_model(vocab_size, embedding_dim, max_length):
    """
    Construct a combined CNN-LSTM model with word embedding
    :param vocab_size: size of the vocabulary
    :param embedding_dim: embedding dimension
    :param max_length: max length of the input sequences
    :return: a keras model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool1D(4),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def deep_learning_model(df):
    """
    Build a deep learning model
    :param df: input data frame containing raw data
    :return: trained neural network
    """
    embedding_dim = 32
    max_length = 200 #np.max([len(news) for news in df['Content'].tolist()])

    # extract data
    X = df[['Title', 'Content']].values
    Y = df['is_fake'].values
    labels = Y.astype('int')

    # tokenize the words
    features, trained_tokenizer = tokenize_words(raw_data=X[:, 1], max_length=max_length)

    # get the size of the vocabulary from the tokenizer
    vocab_size = len(trained_tokenizer.word_index)

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # build the model:
    model = build_nn_model(vocab_size, embedding_dim, max_length)
    #model = build_lstm_model2(vocab_size, embedding_dim, max_length)
    #model = build_gru_model(vocab_size, embedding_dim, max_length)
    #model = build_bidirectional_lstm_model(vocab_size, embedding_dim, max_length)
    model = build_combined_cnn_lstm_model(vocab_size, embedding_dim, max_length)

    # compile the model
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision_m, recall_m, f1_m])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # create training callbacks
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    # train the model
    num_epoch = 100
    history = model.fit(X_train, Y_train,
                        epochs=num_epoch,
                        validation_data=(X_test, Y_test),
                        callbacks=[early_stop_cb, reduce_lr_cb]
                        )

    # print the evaluation results on the test set
    print(evaluate(model, X_test, Y_test, True))

    # visualize the training history
    visualize_dl_training(history)

    # plot the model
    plot_model(model, to_file="output/model_architecture.png")

    # visualize the word embedding
    visualize_trained_word_embedding(model, trained_tokenizer, vocab_size)

    return {
        "fit": model,
        "tokenizer": trained_tokenizer,
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "max_length": max_length
    }


def make_prediction(model_pack, file_path: str, model_name: str):
    """
    Make prediction for a single news file
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
    MAX_SEQUENCE_LENGTH = 300
    MAX_VOCAB = 10000
    x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.is_fake, test_size=0.2)
    tokenizer = Tokenizer(num_words=total_words)

    # update internal vocabulary based on a list of tests
    tokenizer.fit_on_texts(x_train)

    # transformation each text into a sequences integer
    train_sequences = tokenizer.texts_to_sequences(x_train)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    padded_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    padded_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded_train, padded_test, y_train, y_test

def build_lstm_model(padded_train, padded_test, total_words, y_train, y_test):
    # dictionary for storing weights
    trained_models = dict()

    sent_leng = 300
    # # create sequential model
    model = Sequential()

    # embedding layer
    model.add(Embedding(total_words, output_dim=100, input_length=sent_leng))

    # Bi-directional RNN/LSTM
    model.add(Bidirectional(LSTM(100)))

    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    # model.add(Dense(512)),
    # model.add(Dropout(0.3)),
    # model.add(Dense(256)),
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    # cp = ModelCheckpoint('model_Rnn.hdf5', monitor='val_acc', verbose=1, save_best_only=True)
    y_train = np.asarray(y_train)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    # model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               optimizer=tf.keras.optimizers.Adam(1e-4),
    #               metrics=['accuracy'])
    #
    # history = model.fit(padded_train, y_train, batch_size=60, epochs=10, validation_split=0.2, shuffle=False, callbacks=[early_stop])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.0000001, verbose=1)

    # def coeff_determination(y_true, y_pred):
    #     from keras import backend as K
    #     SS_res = K.sum(K.square(y_true - y_pred))
    #     SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    #     return (1 - SS_res / (SS_tot + K.epsilon()))
    #
    # model.compile(loss='mse',
    #               optimizer='nadam',
    #               metrics=[coeff_determination, 'mse', 'mae', 'mape'])

    history = model.fit(padded_train, y_train, validation_data=(padded_test, y_test), epochs=10, batch_size=60, shuffle=False, verbose=1, callbacks=[reduce_lr])
    trained_models['lstm'] = history.history

    # Write the results to .csv files
    # trained_models.to_csv('output/lstm_train_results.csv')

    # visualize the training history
    visualize_dl_training(history)

    return model


def predict_lstm_model(model, padded_test, y_test):
    # create a data frame to store validation metrics and test metrics
    metrics = {"Metrics": ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']}
    test_metrics_df = pd.DataFrame.from_dict(metrics)
    test_metrics_df = pd.DataFrame.from_dict(metrics)

    pred = model.predict(padded_test)
    prediction = []
    # if the predicted value is > 0.5 it is real else it is fake
    for i in range(len(pred)):
        if pred[i] > 0.5:
            prediction.append(1)
        else:
            prediction.append(0)

    # getting the measurement
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1score = f1_score(y_test, prediction)
    auc = roc_auc_score(y_test, prediction)

    # pack the results into a data frame
    values = {
        "Value": [accuracy,
                  precision,
                  recall,
                  f1score,
                  auc]
    }
    test_metrics = pd.DataFrame.from_dict(values)
    test_metrics_df['lstm'] = test_metrics["Value"]

    # Write the results to .csv files
    test_metrics_df.to_csv('output/lstm_test_results.csv')

    print("LSTM Model Accuracy: ", accuracy)
    print("LSTM Model Precision: ", precision)
    print("LSTM Model Recall: ", recall)
    print("LSTM Model F1_score: ", f1score)
    print("LSTM Model AUC: ", auc)

    return prediction

def create_lstm_predictive_model(df):
    df_clean, stop_words, total_words, token_maxlen = process_feature_engineering(df)
    # visualize_fake_word_cloud_plot(df_clean, stop_words)
    # visualize_ligit_word_cloud_plot(df_clean, stop_words)
    padded_train, padded_test, y_train, y_test = create_pad_sequence(df_clean, total_words, token_maxlen)
    model = build_lstm_model(padded_train, padded_test, total_words, y_train, y_test)
    prediction = predict_lstm_model(model, padded_test, y_test)
    visualize_confusion_matrix(prediction, y_test)
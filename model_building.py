import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

def decision_tree_model(df):
    """

    :return:
    """
    # extract data
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    Y = Y.astype('int')

    # extract features
    cv_title = CountVectorizer(max_features=5000)
    mat_title = cv_title.fit_transform(X[:, 0]).todense()
    cv_content = CountVectorizer(max_features=5000)
    mat_content = cv_content.fit_transform(X[:, 1]).todense()

    X_mat = np.hstack((mat_title, mat_content))

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X_mat, Y, test_size=0.2, random_state=0)

    # model
    model = DecisionTreeClassifier()
    fit = model.fit(X_train, Y_train)

    # prediction
    Y_pred = fit.predict(X_test)
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
    print("Precision:", precision_score(Y_test, Y_pred))
    print("Recall:", recall_score(Y_test, Y_pred))
    print("F-Score:", f1_score(Y_test, Y_pred))

    return {
        "fit": fit,
        "title vectorizer": cv_title,
        "content vectorizer": cv_content
    }


def make_prediction(pack, file_path):

    print("Make prediction for", file_path)
    fit = pack['fit']
    cv_title = pack['title vectorizer']
    cv_content = pack['content vectorizer']

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
    mat_title = cv_title.transform(title).todense()
    mat_content = cv_content.transform(content).todense()
    mat_sample = np.hstack((mat_title, mat_content))

    # make prediction
    pred = fit.predict(mat_sample)

    if pred[0] == 1:
        print("This is fake news")
    else:
        print("This is legit news")

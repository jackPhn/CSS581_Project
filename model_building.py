import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score, recall_score,
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


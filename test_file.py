# Import packages
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import requests
import io

# Import data
train_url = "https://raw.githubusercontent.com/selva86/datasets/master/iris_train.csv"
test_url = "https://raw.githubusercontent.com/selva86/datasets/master/iris_test.csv"
s1 = requests.get(train_url).content
s2 = requests.get(test_url).content
training = pd.read_csv(io.StringIO(s1.decode('utf-8')))
test = pd.read_csv(io.StringIO(s2.decode('utf-8')))

# Create the X, Y, training and test
xtrain = training.drop('Species', axis=1)
ytrain = training.loc[:, 'Species']
xtest = test.drop('Species', axis=1)
ytest = test.loc[:, 'Species']

# Init the Gaussian Classifier
model = GaussianNB()

# Train the model
model.fit(xtrain, ytrain)

# Predict output
pred = model.predict(xtest)

# Plot Confusion Matrix
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()

import random
import matplotlib as plt
import pandas as pd
import sklearn

# open csv file
df = pd.read_csv('data.csv')

# print first 5 rows
print(df.head())

# filter out all rows with missing data in any column
df = df.dropna()

# declare features and target
features = df[['x1', 'x2', 'x3']]
target = df['y']

# split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# import model -> decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

# train model
model.fit(X_train, y_train)

# make predictions
predictions = model.predict(X_test)

# evaluate model
from sklearn.metrics import accuracy_score
#print training and testing accuracy
print("Training Accuracy: ", accuracy_score(y_train, model.predict(X_train)))
print("Testing Accuracy: ", accuracy_score(y_test, predictions))
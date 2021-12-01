

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

passengers = pd.read_csv("passengers.csv")
print(passengers.head())

passengers["Sex"] = passengers["Sex"].map({'female': '1', 'male': '0'})
print(passengers.head())

# fill nan values in age column
passengers['Age'].fillna(value=passengers['Age'].mean(), inplace=True)
print(passengers['Age'].values)

#create first class column
passengers['FirstClass'] = passengers['Pclass].apply(lambda p: 1 if p ==1 else 0)

#create second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda p: 1 if p == 2 else 0)


features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived]

train_features, test_features, train_labels, test_labels = train_test_split(features, survival, test_size = 0.2)

#scale feature data
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

#create and train model
model = LogisticRegression()
model.fit(train_features, train_labels)

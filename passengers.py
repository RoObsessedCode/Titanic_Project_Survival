

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

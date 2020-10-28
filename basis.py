import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

data = pd.read_csv('house_prises_final.csv')
data = pd.get_dummies(data)
X = data.drop('SalePrice', axis = 1)
y = data['SalePrice']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(x_train, y_train)

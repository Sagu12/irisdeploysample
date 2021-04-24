# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:55:42 2021

@author: Sagnik.Banerjee
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder


iris= sns.load_dataset("iris")

X= iris.drop("species", axis=1)

y= iris.species

iris.species.value_counts()

lr= LabelEncoder()

y= lr.fit_transform(y.values)


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)

model= GaussianNB()

model.fit(X_train,y_train)

y_pred= model.predict(X_test)

model.score(X_test,y_test)

model.score(X_train, y_train)


sav= joblib.dump(model, "iris.pkl")
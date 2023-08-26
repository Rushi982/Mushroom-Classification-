# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 18:20:11 2023

@author: DELL
"""

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle as pkl

data = pd.read_csv("mushrooms.csv")

print(data.head())

data = data.drop('veil-type', axis=1)

le=LabelEncoder()
for column in data.columns:
    data[column]=le.fit_transform(data[column])


x = data[['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-color','ring-number','ring-type','spore-print-color','population','habitat']]

y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

rfmodel = RandomForestClassifier()

rfmodel.fit(x_train, y_train)


pkl.dump(rfmodel,open('model.pkl', 'wb'))
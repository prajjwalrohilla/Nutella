import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.externals import joblib
df = pd.read_csv("datafinal.csv")
inputs = df
target = df['Quality']
inputs_n = inputs.drop(['S.NO','Time','PM','Quality','Unnamed: 0'],axis='columns')
from sklearn import tree
model = tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs_n,target,test_size=0.4)

model = tree.DecisionTreeClassifier()

model.fit(x_train,y_train)

model.predict(x_test)
model.score(x_test,y_test)
joblib.dump(model,'decisionmodel_joblib')



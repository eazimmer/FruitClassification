import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

titles = ["Diameter", "Weight", "Red" , "Green", "Blue"]
#Read In the File
df = pd.read_csv("citrus.csv")
y = df["name"].values
X = df.drop("name", axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Decision Tree Accuracy
estimator = DecisionTreeClassifier()
estimator.fit(X_train, y_train)
importances = estimator.feature_importances_
y_pred = estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
for index in range(len(importances)):
    print(titles[index] + ": " + str(importances[index]))
print('DecisionTree Classifier: {:.3f}\n'.format(accuracy))

#Decision Tree Forests Accuracy
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators=25,random_state=2)
rf.fit(Xf_train, yf_train)
importancesf = rf.feature_importances_
y_predF = rf.predict(Xf_test) 
accuracyF = accuracy_score(yf_test, y_predF)
for index in range(len(importancesf)):
    print(titles[index] + ": " + str(importancesf[index]))

print('DecisionTreeForest Classifier: {:.3f}\n'.format(accuracyF))

#K Nearest Neighbors Accuracy
Xk_train, Xk_test, yk_train, yk_test = train_test_split(X, y, stratify=y)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X,y)
print('KNN Classifier: {:.3f}'.format(knn.score(Xk_test,yk_test)))
import pandas as pd
import numpy as np
names=["BI_RADS", "age", "shape", "margin", "density","severity"]
file=pd.read_csv(r"mammographic_masses.data.txt",sep=',',na_values=["?"],names=names)
file

file.isnull().sum()

file.describe()

file.dropna(inplace=True)
file

x=file.drop(["BI_RADS",'severity'],axis=1)
y=file['severity']

from sklearn.model_selection import cross_val_score, train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# Decision Trees
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
predict=model.predict(X_test)
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn import tree
from pydotplus import graph_from_dot_data

feature_names=["age", "shape", "margin", "density"]
dot_data = StringIO()
tree.export_graphviz(model, out_file=dot_data,feature_names=feature_names)
graph = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predict)

#Cross Validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model,x,y, cv=10)

print(scores.mean())
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)
predict=model.predict(X_test)
accuracy_score(y_test,predict)

#SVM
from sklearn import svm, datasets
model = svm.SVC(kernel='linear', C=1.0)
model.fit(X_train,y_train)
predict=model.predict(X_test)
accuracy_score(y_test,predict)

#KNN
from sklearn .neighbors import KNeighborsClassifier
from math import sqrt
sqrt(len(y_test))
model=KNeighborsClassifier(n_neighbors=13, p=2,metric='euclidean')
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print(accuracy_score(y_test,predictions))

# Naive Bayes
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB

scaler = preprocessing.MinMaxScaler()
x_minmax = scaler.fit_transform(x)

model = MultinomialNB()
cv_scores = cross_val_score(model, x_minmax,y, cv=10)

cv_scores.mean()


# Revisiting SVM
#sigmoid Kernal
model = svm.SVC(kernel='sigmoid', C=1.0)
model.fit(X_train,y_train)
predict=model.predict(X_test)
accuracy_score(y_test,predict)

#poly Kernal
model = svm.SVC(kernel='poly', C=1.0)
model.fit(X_train,y_train)
predict=model.predict(X_test)
accuracy_score(y_test,predict)

#rbf Kernal
model = svm.SVC(kernel='rbf', C=1.0)
model.fit(X_train,y_train)
predict=model.predict(X_test)
accuracy_score(y_test,predict)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
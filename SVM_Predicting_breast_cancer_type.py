import numpy as np
from sklearn import preprocessing, model_selection, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace = True)
df.drop('id', 1, inplace=True)

X = np.array(df.drop('class',1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accurary = clf.score(X_test, y_test)
print(accurary)

examples_measures = np.array(
    [
    [4,2,1,1,1,2,3,2,1],
    [4,2,1,2,3,2,3,2,1]
    ]
)

examples_measures = examples_measures.reshape(len(examples_measures),-1)

prediction = clf.predict(examples_measures)
print(prediction)

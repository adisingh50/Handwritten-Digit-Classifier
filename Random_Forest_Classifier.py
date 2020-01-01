import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('train.csv').as_matrix()

print(data)
print(data.shape)

clf = RandomForestClassifier()

#train round #1
xtrain = data[0:30000, 1:] #we take first half of the rows and exclude the first column (the digit)
train_label = data[0:30000, 0] #take the labels for the first half
clf.fit(xtrain, train_label)

xtest = data[30000:, 1:]   #taking the second half of the rows for testing data
test_label = data[30000:, 0]

print('Predicted Number: ')
print(clf.predict([xtest[1234]]))

print('Actual Number: ')
print(test_label[1234])

pr = clf.predict(xtest)

count = 0
for i in range(0, 12000):
    count +=1 if pr[i] == test_label[i] else 0
print("Accuracy: ", count/12000)

#saving the RF Classifier to local drive
import pickle

pickle.dump(clf, open('nr_model','wb'))

import numpy as np
from sklearn.svm import SVC
from sklearn import svm

from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
#files
from preprocessing import readfile, calc_length, flatten
from preprocessing import padding_data, one_hot_encoding
from preprocessing import CHANNELS
from sklearn.metrics import classification_report

folder = "data"
train_data = readfile(f"{folder}/ae.train", 0)
test_data = readfile(f"{folder}/ae.test", 1)
maxlength = calc_length(test_data, train_data)

train_input, train_output = padding_data(train_data, maxlength, 0, True)
test_input, test_output = padding_data(test_data, maxlength, 1, True)

"""
Linear classifiers SGDC
(Using default loss=hinge gives a linear SVM)
accuracy 5 folds: 0.9702702702702702
accuracy 50 folds: 0.9783783783783784
"""
cv = KFold(n_splits=5, random_state=42, shuffle=True)
sgdc = SGDClassifier(alpha = 0.03, max_iter = 1000)
scores = []
test_output = test_output.ravel()
train_output = train_output.ravel()
for train_index, test_index in cv.split(train_input):
    X_train, y_train = train_input[train_index] , train_output[train_index]
    X_test, y_test = train_input[test_index] , train_output[test_index]
    sgdc.fit(X_train, y_train)
    scores.append(sgdc.score(X_test, y_test))

test_scores = sgdc.score(test_input, test_output)
pred = sgdc.predict(test_input)
print(confusion_matrix(pred, test_output))
print(test_scores)

#classifier = svm.SVC(kernel="rbf", gamma='auto', C=100000)
#classifier.fit(X_train, y_train)

#y_predict = classifier.predict(X_test)
#print(classification_report(y_test, y_predict))
model = SVC(kernel="rbf", gamma="auto", C=50)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, datasets
from sklearn.datasets import load_digits

digits = load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=False)

clf = svm.SVC(kernel='rbf', gamma=0.001)

clf.fit(X_train, y_train)

predicted = clf.predict(X_test)


print(f"Classification report for SVM classifier:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")
print(f"Accuracy Score: {metrics.accuracy_score(y_test, predicted) * 100:.2f}%")
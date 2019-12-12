from sklearn import svm
import pickle
import numpy as np
import sys

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

import time


start = time.time()

accuracy = 0
pickleFilePath = "D:\\cohn-kanade\\cohn_dataset_svm.p"
# pickleFilePath = "C:\\Users\\vince\\Desktop\\cohn-kanade\\cohn_dataset_svm.p"

(training_data, validation_data, test_data) = pickle.load(open(pickleFilePath, 'rb'))
(X_train, y_train), (X_test, y_test) = (training_data[0], np.ravel(training_data[1])), (test_data[0], np.ravel(test_data[1]))

print('Number of feature:', len(X_train[0]))
print("X_train lenght:", len(X_train))
print("y_train lenght:", len(y_train))
print("X_test lenght:", len(X_test))
print("y_test lenght:", len(y_test))
print()

# RBF
clf = svm.SVC(C=8.85,
              gamma=0.23,
              decision_function_shape='ovo')
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
score = clf.score(X_test, y_test)
print('RBF score:', score)

end = time.time()
print('Total time:', end - start)

accuracy = 0
print()

# LINEAR
'''
lin_clf = svm.LinearSVC(C=1.43)
lin_clf.fit(X_train, y_train)
prediction = lin_clf.predict(X_test)
score = lin_clf.score(X_test, y_test)
print('Linear score:', score)
'''

# Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


np.set_printoptions(precision=2)
plot_labels = ['Neutral', 'Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
cm = confusion_matrix(y_test, prediction)
plt.figure()
plot_confusion_matrix(cm, plot_labels,
                      title='Confusion matrix, without normalization')
plt.figure()
plot_confusion_matrix(cm, plot_labels,
                      normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# Grid search for best hyper parameters LINEAR
'''
Best_score_lin = 0
C = np.logspace(-3, 2, 20)
for c in C:
        lin_clf = svm.LinearSVC(C=c)
        lin_clf.fit(X_train, y_train)
        score = lin_clf.score(X_test, y_test)
        print("C = {} & score = {}".format(c, score))
        sys.stdout.flush()
        if(score > Best_score_lin):
            Best_score_lin = score
            Best_C_lin = c
print(Best_C_lin, Best_score_lin)
print('Done!')
'''

# Grid search for best hyper parameters RBF
'''
Best_score = 0
c = np.logspace(-3, 2, 20)
gamma = np.logspace(-3, 2, 20)
print(len(c)*len(gamma))
for C in c:
    for G in gamma:
        clf = svm.SVC(gamma=G, C=C)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("C = {} & G = {} & score = {}".format(C, G, score))
        sys.stdout.flush()
        if(score > Best_score):
            Best_score = score
            Best_C = C
            Best_G = G
print(Best_C, Best_G, Best_score)
print('Done!')
'''

# Grid search answers
'''
print()
print("RBF grid search:")
print(Best_C, Best_G, Best_score)
print("Linear grid search:")
print(Best_C_lin, Best_score_lin)
'''

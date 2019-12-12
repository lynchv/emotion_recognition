import pickle
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

import time


start = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Removing warnings
# np.random.seed(1337)  # for reproducibility
pickleFilePath = "D:\\cohn-kanade\\cohn_dataset_cnn.p"

batch_size = 30
nb_classes = 8
nb_epoch = 1

# input image dimensions
img_rows, img_cols = 100, 100
# number of convolutional filters to use
nb_filters1 = 32
nb_filters2 = 64
nb_filters3 = 128
nb_filters4 = 256
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(training_data, validation_data, test_data) = pickle.load(open(pickleFilePath, 'rb'))
(X_train, y_train), (X_test, y_test) = (training_data[0], training_data[1]), (test_data[0], test_data[1])

# Ckecks if backend is theano or tensorflow for dataset format
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# A sequential model (feedforward)
print("Starting to build Model")
model = Sequential()

# adding 2 Convolutional Layers and a maxpooling layer with activation function rectified linear unit and  Dropout for regularization
# Filter stride is default to 1
# LAYER 1
model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))  # Stride default to pool_size, 2x2
# LAYER 2
model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
# LAYER 3
model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
# LAYER 4
model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

# Fully Conntected Layers with relu and a output layer with softmax
model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# Training
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))

# Testing
probability = model.predict(X_test, verbose=0)
classes = model.predict_classes(X_test, verbose=0)
score = model.evaluate(X_test, Y_test, verbose=0)

end = time.time()
print('Total time:', end - start)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Précision du model en fonction des époques
'''
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

# Model summary
# model.summary()

# Matrice de confusion
'''
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

    print(cm)

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
cm = confusion_matrix(np.ravel(y_test), classes)
plt.figure()
plot_confusion_matrix(cm, plot_labels,
                      title='Confusion matrix, without normalization')
plt.figure()
plot_confusion_matrix(cm, plot_labels,
                      normalize=True,
                      title='Normalized confusion matrix')

plt.show()
'''

## https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.

'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LocallyConnected2D
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.optimizers import Adam
from TestTools import plot_results, show_misclassified, load_data

batch_size = 128
num_classes = 10
epochs = 15
img_rows, img_cols = 16, 16

loss_results = []
accuracy_results = []

for i in range(0,10):
	# input image dimensions

	x_train, y_train, x_test, y_test = load_data(img_rows, img_cols)

	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

	model = Sequential()
	model.add(LocallyConnected2D(32, (3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(LocallyConnected2D(64, (3, 3), activation='tanh'))
	model.add(Flatten())
	model.add(Dense(128, activation='sigmoid'))
	model.add(Dense(num_classes, activation='softmax'))

	# learning is very slow
	# model.compile(loss='categorical_crossentropy',
	#              optimizer=Adam(lr=0.00005),
	#              metrics=['accuracy'])

	# learning is too fast
	# model.compile(loss='categorical_crossentropy',
	#               optimizer=Adam(lr=0.05),
	#               metrics=['accuracy'])

	# learning is effective
	model.compile(loss='categorical_crossentropy',
	             optimizer=Adam(lr=0.001),
	             metrics=['accuracy'])

	history = model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=2,
	          validation_data=(x_test, y_test))

	score = model.evaluate(x_test, y_test, verbose=0)
	print("Test Run: " , i)
	print()
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	print()
	loss_results.append(score[0])
	accuracy_results.append(score[1])

	model = None

loss_results = pd.Series(loss_results)
accuracy_results = pd.Series(accuracy_results)

print("Loss Statistics")
print(loss_results.describe())
print()
print("Accuracy Statistics")
print(accuracy_results.describe())

# plot_results(history,epochs)

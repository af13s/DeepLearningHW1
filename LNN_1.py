## https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LocallyConnected1D
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.optimizers import Adam

batch_size = 32
num_classes = 10
epochs = 15

# input image dimensions
img_rows, img_cols = 16, 16

def plot_results(history, epochs):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
	plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, epochs), history.history["acc"], label="train_acc")
	plt.plot(np.arange(0, epochs), history.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="upper left")
	plt.show()

def load_data():

	trainData = []
	trainLabels = []
	testData = []
	testLabels = []

	classes = [0,1,2,3,4,5,6,7,8,9]
	label_encoder = pd.factorize(classes)
	labels_1hot = OneHotEncoder().fit_transform(label_encoder[0].reshape(-1,1))
	onehot_array = labels_1hot.toarray()

	d1 = dict(zip(classes,onehot_array.tolist()))

	for i,file in enumerate(['zip_train.txt', 'zip_test.txt']):
		f = open(file)

		tempLabels = []
		tempData = []

		for line in f:
			line = line.split()
			thelabel = line.pop(0)
			thelabel = int(float((thelabel)))

			line = [float(i) for i in line]
			theX = np.interp(line, (-1,1), (0, 1))

			tempLabels.append(thelabel)
			tempData.append(theX)

		for label in tempLabels:
		    encoding = d1[label]
		    if i == 0:
		    	trainLabels.append(encoding)
		    else:
		    	testLabels.append(encoding)

		if i == 0:
			trainLabels = np.array(trainLabels).reshape((-1,len(classes)))
			trainData = np.array(tempData)
		else:
			testLabels = np.array(testLabels).reshape((-1,len(classes)))
			testData = np.array(tempData)

		f.close()

	return trainData, trainLabels, testData, testLabels


x_train, y_train, x_test, y_test = load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
input_shape = (img_rows, img_cols)


model = Sequential()
model.add(LocallyConnected1D(32, (3),
                 activation='relu',
                 input_shape=input_shape))
model.add(LocallyConnected1D(64, (3), activation='tanh'))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plot_results(history,epochs)
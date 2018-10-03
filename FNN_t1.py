import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

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

if __name__ == "__main__":

	trainData, trainLabels, testData, testLabels = load_data()

	batch_size = 128
	num_classes = 10
	epochs = 20

	model = Sequential()
	model.add(Dense(167, activation='relu', input_shape=(256,)))
	model.add(Dropout(0.2))
	model.add(Dense(167, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',
	              optimizer=Adam(lr=0.001),
	              metrics=['accuracy'])

	history = model.fit(trainData, trainLabels,
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    verbose=1,
	                    validation_data=(testData,testLabels))

	score = model.evaluate(testData, testLabels, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	plot_results(history,epochs)
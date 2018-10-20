import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Nadam
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
import pandas as pd
from TestTools import plot_results, show_misclassified

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

if __name__ == "__main__":

	trainData, trainLabels, testData, testLabels = load_data()

	batch_size = 1
	num_classes = 10
	epochs = 10

	# # Parameter Intialization
	# model = Sequential()
	# model.add(Dense(128, activation='relu', bias_initializer=keras.initializers.Constant(value=0.01), input_shape=(256,)))
	# model.add(Dense(128, bias_initializer=keras.initializers.Constant(value=6.7), activation='sigmoid'))
	# model.add(Dense(128, bias_initializer=keras.initializers.Constant(value=6.7), activation='tanh'))


	model = Sequential()
	model.add(Dense(128, activation='relu', input_shape=(256,)))
	model.add(BatchNormalization())
	model.add(Dense(128, activation='sigmoid'))
	model.add(BatchNormalization())
	model.add(Dense(128, activation='tanh'))
	model.add(BatchNormalization())


	model.add(Dense(num_classes, activation='softmax'))

	model.summary()

	# # Learning Rate
	# model.compile(loss='categorical_crossentropy',
	#               optimizer=Adam(lr=0.05),
	#               metrics=['accuracy'])

	model.compile(loss='categorical_crossentropy',
	              optimizer=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999),
	              metrics=['accuracy'])

	# model.compile(loss='categorical_crossentropy',
 #              optimizer=Adam(lr=0.001),
 #              metrics=['accuracy'])

	#print(model.get_weights())
	
	history = model.fit(trainData, trainLabels,
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    verbose=2,
	                    validation_data=(testData,testLabels))

	score = model.evaluate(testData, testLabels, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	plot_results(history,epochs)

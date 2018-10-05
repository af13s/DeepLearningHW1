import numpy as np
import matplotlib.pyplot as plt

def show_misclassified(test_set, model):

	tempLabels = []

	for i,file in enumerate(['zip_test.txt']):
		f = open(file)

		tempData = []

		for line in f:
			line = line.split()
			thelabel = line.pop(0)
			thelabel = int(float((thelabel)))

			line = [float(i) for i in line]
			theX = np.interp(line, (-1,1), (0, 1))

			tempLabels.append(thelabel)
			tempData.append(theX)

	misclassified = {}

	
	predictions = model.predict_classes(test_set)
	print (predictions)

	print(len(predictions))
	print(len(tempLabels))

	for i in range(len(predictions)):
		if predictions[i] != tempLabels[i]:
			print(predictions[i] , " was classified as " , tempLabels[i] )
			if tempLabels[i] not in misclassified:
				misclassified[tempLabels[i]] = 0 
			misclassified[tempLabels[i]] += 1

	print (misclassified)

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
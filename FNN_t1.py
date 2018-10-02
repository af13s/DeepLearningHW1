import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

f = open('zip_train.txt')

trainLabels = []
trainData = []

for line in f:
	line = line.split()
	thelabel = line.pop(0)
	thelabel = int(float((thelabel)))

	line = [float(i) for i in line]
	theX = np.interp(line, (-1,1), (0, 1))

	trainLabels.append(thelabel)
	trainData.append(theX)

	# For visualization
	# pixels = theX.reshape((16, 16))

	# plt.title('Label is {thelabel}'.format(thelabel=thelabel))
	# plt.imshow(pixels, cmap='gray')
	# plt.show()

classes = [0,1,2,3,4,5,6,7,8,9]
label_encoder = pd.factorize(classes)
encoder = OneHotEncoder()
labels_1hot = encoder.fit_transform(label_encoder[0].reshape(-1,1))
onehot_array = labels_1hot.toarray()

d1 = dict(zip(classes,onehot_array.tolist()))
theFinalLabels = []
for aminoacid in trainLabels:
    encoding = d1[aminoacid]
    theFinalLabels.append(encoding)

nicelabels = np.array(theFinalLabels)
nicelabels = nicelabels.reshape((-1,len(classes)))

model = Sequential()
model.add(Dense(167, activation='relu', input_shape=(256,)))
model.add(Dropout(0.2))
model.add(Dense(167, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

batch_size = 128
num_classes = 10
epochs = 20

trainData = np.array(trainData)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

history = model.fit(trainData, nicelabels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)

score = model.evaluate(trainData, nicelabels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show()

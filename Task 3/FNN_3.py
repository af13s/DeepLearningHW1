import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
from keras import regularizers
from TestTools import load_data, plot_results

if __name__ == "__main__":

        loss_results = []
        accuracy_results = []
        img_rows, img_cols = 16, 16


        for i in range(0,10):
            trainData, trainLabels, testData, testLabels = load_data(img_rows, img_cols)

            batch_size = 128
            num_classes = 10
            epochs = 50 

            model = Sequential()
            model.add(Dense(128, activation='relu', input_shape=(256,), kernel_regularizer=regularizers.l1(0.001)))
            model.add(Dense(128, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001)))
            model.add(Dense(128, activation='tanh', kernel_regularizer=regularizers.l1(0.001)))
            model.add(Dense(num_classes, activation='softmax'))

            model.summary()

            model.compile(loss='categorical_crossentropy',
                        optimizer=Adam(),
                        metrics=['accuracy'])

            history = model.fit(trainData, trainLabels,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=(testData,testLabels))

            score = model.evaluate(testData, testLabels, verbose=0)
            print("Test Run: " , i)
            print()
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            print()
            loss_results.append(score[0])
            accuracy_results.append(score[1])

            model = None
    
            #plot_results(history,epochs)

        loss_results = pd.Series(loss_results)
        accuracy_results = pd.Series(accuracy_results)
        
        print("Loss Statistics")
        print(loss_results.describe())
        print()
        print("Accuracy Statistics")
        print(accuracy_results.describe())
        '''
        classes = ['0','1','2','3','4','5','6','7','8','9']
        predictions = model.predict(testData).argmax(axis=-1)
        trueOutputs = testLabels.argmax(axis=-1)
        cm = confusion_matrix(trueOutputs,predictions)
        plot_confusion_matrix(cm,classes)
        '''

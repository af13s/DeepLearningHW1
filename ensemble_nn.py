# https://towardsdatascience.com/ensembling-convnets-using-keras-237d429157eb

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LocallyConnected1D
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.optimizers import Adam
from TestTools import plot_results, show_misclassified, load_data
import numpy as np

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Average, Dropout
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

img_rows, img_cols = 16,16
num_classes = 10
batch_size = 32 
epochs = 5

x_train, y_train, x_test, y_test = load_data(img_rows, img_cols)



def cnn_data(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    return x_train, y_train, x_test, y_test, input_shape

def lnn_data(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
    input_shape = (img_rows, img_cols)

    return x_train, y_train, x_test, y_test, input_shape

fnn_dataset = (x_train, y_train, x_test, y_test, (256,))
cnn_dataset = cnn_data(x_train, y_train, x_test, y_test)
lnn_dataset = lnn_data(x_train, y_train, x_test, y_test)


# Ensemble of Networks

def cnn( dataset ):

    x_train, y_train, x_test, y_test, input_shape = dataset

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='tanh'))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def cnn2( dataset ):

    x_train, y_train, x_test, y_test, input_shape = dataset

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(6, 6),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (6, 6), activation='tanh'))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def cnn3( dataset ):

    x_train, y_train, x_test, y_test, input_shape = dataset

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def compile_and_train(model, dataset, epochs): 

    x_train, y_train, x_test, y_test, input_shape = dataset
    
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

    return history

#Initialize network input
cnn_model = cnn(cnn_dataset)
cnn_model_2 = cnn2(cnn_dataset)
cnn_model_3 = cnn3(cnn_dataset)

cnn_model_ = cnn(cnn_dataset)
cnn_model_2_ = cnn2(cnn_dataset)
cnn_model_3_ = cnn3(cnn_dataset)

#train network
compile_and_train(cnn_model, cnn_dataset, epochs = epochs)
compile_and_train(cnn_model_2, cnn_dataset, epochs = epochs)
compile_and_train(cnn_model_3, cnn_dataset, epochs = epochs)

compile_and_train(cnn_model_, cnn_dataset, epochs = epochs)
compile_and_train(cnn_model_2_, cnn_dataset, epochs = epochs)
compile_and_train(cnn_model_3_, cnn_dataset, epochs = epochs)

#evaluate loss
# evaluate_error(cnn_model)
# evaluate_error(fnn_model)
# evaluate_error(lnn_model)

# # Three Model Ensemble
# conv_pool_cnn_model = conv_pool_cnn(model_input)
# all_cnn_model = all_cnn(model_input)
# nin_cnn_model = nin_cnn(model_input)

# conv_pool_cnn_model.load_weights('weights/conv_pool_cnn.29-0.10.hdf5')
# all_cnn_model.load_weights('weights/all_cnn.30-0.08.hdf5')
# nin_cnn_model.load_weights('weights/nin_cnn.30-0.93.hdf5')
 
# models = [cnn_model, fnn_model, lnn_model]
models = [cnn_model, cnn_model_2, cnn_model_3, cnn_model_, cnn_model_2_, cnn_model_3_]

def ensemble(models):

    count = 0
    predictions = None
    
    predictions =  np.array(models[0].predict(cnn_dataset[2]))
    predictions +=  np.array(models[1].predict(cnn_dataset[2]))
    predictions +=  np.array(models[2].predict(cnn_dataset[2]))
    predictions +=  np.array(models[3].predict(cnn_dataset[2]))
    predictions +=  np.array(models[4].predict(cnn_dataset[2]))
    predictions +=  np.array(models[5].predict(cnn_dataset[2]))
    # predictions +=  np.array(models[1].predict(fnn_dataset[2]))
    # predictions +=  np.array(models[2].predict(lnn_dataset[2]))

    predictions = (predictions/len(models))

    for i in range(len(predictions)):
        correct_class = y_test[i].argmax()
        guess = predictions[i].argmax()
        if guess == correct_class:
            count+=1
    
    print ("\nEnsemble Accuracy: " , str(count/len(y_test)))

ensemble(models)

# DeepLearningHW1
Neural Network Design and Optimization

## Task I – Neural Network Design
In the deep learning framework you have established, design three different neural networks. Each one must
have at last four layers; rectified activation functions must be used at least in one of the layers (among all the
three networks) and sigmoid/tanh activation functions must be used in the one of the layers as well (among
all the three networks). The overall connection requirements are as follows:

(1) Fully connected, where each input/neuron is connected to all the neurons in the next layer

(2) Locally connected with no weights shared in the first three layers, where each input/neuron is
connected to the neurons in a local neighbor in the next layer

(3) Locally connected with weights shared in the first three layers (i.e., a convolutional neural network)
In your report, you need to provide explanations of your design choices and describe your neural networks
clearly.

## Task II - Techniques for Optimization
You need to do the required analysis and then perform experiments for each of the three networks.

(1) Parameter initialization strategies. For each of the networks, analyze how parameters should be
initialized. Then demonstrate three cases based on your analysis: 1) learning is very slow; 2) learning
is effective (i.e., fast with accurate results); and 3) the learning is too fast (i.e., the network does not
give good performance).

(2) Learning rate. Estimate a good learning rate for each of the networks. Then demonstrate three cases
based on your analysis: 1) learning is very slow; 2) learning is effective; and 3) learning is too fast.

(3) Explain how the batch size would impact the batch normalization for each of the networks. Then
demonstrate an effective batch size and an ineffective batch size on each of the three networks you
have.

(4) Momentum. Commonly used momentum coefficient values are 0.5, 0.9, and 0.99. Using the best
parameter initialization strategy, the best learning rate, and the best batch size you have found so far, 
experiment with the three different momentum values on the three networks you have and document
the results. Explain the differences you have observed on the three neural networks you have.

## Task III - Techniques for Improving Generalization
For this task, you need to do the required analysis and apply the following regularization techniques with the
goal to improve the performance on the 2007 samples in zip_test.txt.

(1) Use an ensemble to improve the generalization performance. Here you need to use bagging of at least
six neural networks to improve the performance of the individual neural networks. You need to analyze
your results.

(2) Dropout. Explain the effects of the dropout parameter (probability of keeping a neuron) on the three
neural networks you have. Then demonstrate an effective case and an ineffective case on each of the
three neural networks you have.

(3) L1 regularization. Explain the effects of the L1 regularization on the three neural networks you have.
Then demonstrate an effective L1 regularization case and an ineffective L1 regularization case on each
of the three neural networks you have.

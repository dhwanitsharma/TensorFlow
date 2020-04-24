This Neural Network is designed to predict clothing items in the fashion_mnist dataset provided in keras.
The Neural network consists of 3 layers as follows :
Layer 1 - Input layer : We use flatten to convert the 28*28 data into vector form of 784
Layer 2 - Hidden layer: Dense denotes that this layer is fully connected and each neuron from the layer 1. 
                        It has 128 neurons and uses rectify linear unit activation func.
Layer 3 - Output layer: Has 10 neurons to represent probability of 10 labels. 
						It uses softmax activation func.             
						


The code can be executed on the google collab link below:
https://colab.research.google.com/drive/1ERNHBcFQPyVPI-aLJOX7FYmIH1WevUYC
'''''''''
Project: Digit recognition using sigmoid neural network using MNIST data from SCRATCH!
Kaleem Ahmad
IIT(ISM) Dhanbad
CSE
'''''''''

import NeuralNetwork
import LoadData

training_data, validation_data, test_data = LoadData.ProcessData()
neuron = NeuralNetwork.SigmoidNeuralNetwork([784, 30, 10])
neuron.SGD(training_data, 25, 10, 3.0, test_data = test_data)
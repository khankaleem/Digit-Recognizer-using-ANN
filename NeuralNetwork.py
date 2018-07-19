'''''''''
This module contains the library for a SigmoidNeuralNetwork using Stochastic Gradient Descent.
'''''''''

#import libraries
import numpy as np 
import random

#sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#derivative of sigmoid function    
def Dsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

#build class
class SigmoidNeuralNetwork():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]#stores bias values for all components of each layer i of size sizes[i]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]#stores weights of all edges from layer i to i+1    
        
    def Dcost(self, output_activations, y):       
        return (output_activations-y)
    
    def Output(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def Evaluate(self, test_data):
        output = [(np.argmax(self.Output(x)), y) for x, y in test_data]
        return (sum((x==y) for x, y in output)/len(test_data))*100.0
    
    #stochastic gradient descent
    def SGD(self,  training_data, epochs, batch_size, eta, test_data = None):
        training_data = list(training_data)
        if test_data:
            test_data = list(test_data)                
        for i in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]
            for batch in batches:
                self.UpdateNetwork(batch, eta)
            '''
            process test_data after epoch
            '''
            if test_data:
                print('Epoch ' + str(i+1) + ': ' + str(self.Evaluate(test_data)))
            else:
                print('Epoch ' + str(i+1))
                
    #update weights and biases by applying backpropagation    
    def UpdateNetwork(self, batch, eta):
        Slopeb = [np.zeros(b.shape) for b in self.biases]
        Slopew = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            slopeb, slopew = self.BackPropogate(x, y)
            Slopew = [w+del_w for w, del_w in zip(Slopew, slopew)]    
        '''
        moidfy weights an biases
        '''
        self.weights = [w-(eta/len(batch))*s for w, s in zip(self.weights, Slopew)]    
        self.biases  = [b-(eta/len(batch))*s for b, s in zip(self.biases, Slopeb)]    
            
            
    #using bakpropagation figure out the changes in each of biase and weight    
    def BackPropogate(self, x, y):
        slopeb = [np.zeros(b.shape) for b in self.biases]
        slopew = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        z = []
        a = [x]
        
        for b, w in zip(self.biases, self.weights):
            activation = np.dot(w, activation)+b
            z.append(activation)
            activation = sigmoid(activation)
            a.append(activation)
            
        delta = self.Dcost(a[-1], y)*Dsigmoid(z[-1]) 
        slopeb[-1] = delta
        slopew[-1] = np.dot(delta, a[-2].transpose())        

        '''
        b changes z which in turn changes activation which changes Cost
        slopb = 1*Dsigmoid(z)*how cost changes wrt a = w*delta
        b changes z which in turntgrftt changes activation which changes Cost
        slopw =  delta*a[previous]       
        '''
        for i in range(2, self.num_layers):
            delta = np.dot(self.weights[-i+1].transpose(), delta) * Dsigmoid(z[-i])
            slopeb[-i] = delta
            slopew[-i] = np.dot(delta, a[-i-1].transpose())
        return (slopeb, slopew)
 
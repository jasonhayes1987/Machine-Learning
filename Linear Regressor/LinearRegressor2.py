import numpy as np
import matplotlib.pyplot as plt

class Linear_Regressor:
    
    def __init__(self):
        self.weights = []
        self.train_costs = []
        self.test_costs = []
        
    def train(self, Xtrain, Ytrain, Xtest, Ytest, learning_rate, epochs):
        # set N to number of training samples and D to number of training features
        N, D = Xtrain.shape
        K = Ytrain.shape[1]

        # store calculated costs so they can be plotted to demonstrate gradient descent
        self.test_costs = []

        # initialize random weights to have a variance of 1/D
        self.weights = np.random.randn(D,K) / np.sqrt(D)

        # code gradient descent algo
        for i in range(epochs):
            Yhat = Xtrain.dot(self.weights) # calculate predictions
            delta = Yhat - Ytrain # calculate delta between prediction and actual value
            self.weights = self.weights - (learning_rate*Xtrain.T.dot(delta)) # update weight
            mse = np.square(delta).mean() # calculate mean squared error
            self.train_costs.append(mse) # append mse to array of costs that will be used for plotting
            
            # calculate error on test set and append to test cost array
            test_yhat = Xtest.dot(self.weights)
            test_delta = test_yhat - Ytest
            test_mse = np.square(test_delta).mean()
            self.test_costs.append(test_mse)
            
            if i%100 == 0:
                print(f'Train Mean Squared Error: {mse}, Test Mean Squared Error: {test_mse}')
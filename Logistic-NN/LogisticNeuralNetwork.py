import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Util import error_rate, y_to_indicator, elu_vectorized, elu_prime_vectorized

class Logistic_Regressor:
    
    def __init__(self, hidden_layers, hidden_nodes, store_metrics=False):
        self.weights= []
        self.bias = []
        self.weight_momentum = []
        self.bias_momentum = []
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.train_costs = []
        self.test_costs = []
        self.weight_cache = []
        self.bias_cache = []
        self.bias_corrected_weight_momentums = []
        self.bias_corrected_bias_momentums = []
        self.bias_corrected_weight_cache = []
        self.bias_corrected_bias_cache = []
        self.activation = ''
        self.error_rate = 0
        self.final_score = 0
        self.store_metrics = store_metrics
        
        if self.store_metrics:
            # arrays created to store all metrics across all epochs
            self.Yhats = []
            self.all_hidden_layers = []
            self.all_weights = []
            self.all_biases = []
            self.all_weight_momentums = []
            self.all_bias_momentums = []
            self.all_weight_caches = []
            self.all_bias_caches = []
            self.all_weight_gradients = []
            self.all_bias_gradients = []
            self.all_Xbatches = []
            self.all_Ybatches = []
            self.all_momentum_corrections = []
            self.all_cache_corrections = []
            self.all_bias_corrected_weight_momentums = []
            self.all_bias_corrected_bias_momentums = []
            self.all_bias_corrected_weight_caches = []
            self.all_bias_corrected_bias_caches = []
            self.all_deltas = []

    
    def forward(self, X, W, b):
        """
        computes one step forward through a neural network and returns array z of hidden layer output

        Input:
        X = inputs
        W = weights
        b = bias
        
        Returns:
        z = hidden layer output
        """
        
        # compute value at hidden node Z
        # relu
        if self.activation == 'relu':
            z = X.dot(W) + b
            z[z < 0] = 0
            
        # elu
        elif self.activation == 'elu':
            z = X.dot(W) + b
            z = elu_vectorized(z, 1)
#             z[z <= 0 ] = np.exp(X.dot(W) + b) - 1
        
        # tanh
        elif self.activation == 'tanh':
            z = np.tanh(X.dot(W) + b)
        
        # sigmoid
        elif self.activation == 'sigmoid':
            z = 1 / (1 + np.exp(-X.dot(W) - b))

        else:
        # print a statement indicating the passed activation is not defined
            print('ACTIVATION UNDEFINED')
        
        return z
    
    
    
    def softmax(self, A):
        """
        Returns the softmax of the passed array
        
        Input:
        A: array to calculate softmax of
        
        Returns:
        y: softmax of 'A'
        """
        # check if passed in array 'A' is one-dimensional and if so, expand to 2 dimensions and transpose
        if len(A.shape) == 1:
            A = np.expand_dims(A, axis=1).T

        exp = np.exp(A)
        y = exp/exp.sum(axis=1,keepdims=True)
    
        return y
    
    
    def cross_entropy(self, T, Y_hat):
        """
        Returns the cross-entropy of targets (T) and predictions (Y_hat)
        
        Input:
        T: {array} target outcomes
        Y_hat: {array} predicted outcomes
        
        Returns:
        cross entropy of 'T' and 'Y_hat'
        """
        return -(T*np.log(Y_hat)).sum()
    
    def cross_entropy2(self, T, Y_hat):
        """
        Returns the cross-entropy of targets (T) and predictions (Y_hat)
        
        Input:
        T: {array} target outcomes
        Y_hat: {array} predicted outcomes
        
        Returns:
        cross entropy of 'T' and 'Y_hat'
        """
        n = len(T)
        return -np.log(Y_hat[np.arange(n), T]).mean()
    
    
    def delta(self, T, Y_hat, Z):
        """
        Returns array of deltas between derivatives to be used in calculating the weight updates across the network
        
        Inputs:
        T: {array} target outcomes
        Y_hat: {array} predicted outcomes
        Z: {array} node (neuron) values at each hidden layer
        
        Returns:
        d: deltas between derivatives of layer gradients
        """
        
        d = []
        d.append(Y_hat - T)

        for l in range(self.hidden_layers):
            # sigmoid
            if self.activation == 'sigmoid':
                d.append(d[l].dot(self.weights[len(self.weights)-l-1].T) * (1 - Z[len(Z)-l-1] * Z[len(Z)-l-1]))
            
            # rulu
            if self.activation == 'relu':
                d.append(d[l].dot(self.weights[len(self.weights)-l-1].T) * (Z[len(Z)-l-1] > 0))
                
            # elu
            if self.activation == 'elu':
                z = elu_prime_vectorized(Z[len(Z)-l-1], 1)
                d.append(d[l].dot(self.weights[len(self.weights)-l-1].T) * z)
        
            # tanh
            if self.activation == 'tanh':
                d.append(d[l].dot(self.weights[len(self.weights)-l-1].T) * (1 - Z[len(Z)-l-1]**2))
        
        d.reverse()

        return d
    
    
    
    def calculate_weight_gradients(self, D, Z, X):
        """
        calculates the gradients of the weights of a neural network

        INPUT:
        D: array of deltas (calculated by delta func)
        Z: array of hidden layer values
        X: array of training data

        RETURN:
        g: gradients for each weight in the neural net
        """
        
        # check if X is one-dimensional and if so, expand dimension and transpose to be 1xN
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1).T
        
        g = []
        g.append(X.T.dot(D[0]))

        for z in range(len(Z)):
            if len(Z[z].shape) == 1:
                Z[z] = np.expand_dims(Z[z], axis=1).T
 
            g.append(Z[z].T.dot(D[z+1]))

        return g
    
    
    
    def update_weights(self, learning_rate, epsilon):
        """
        updates weight values of neural net

        INPUT:
        learning_rate: learning rate to multiply gradient by
        epsilon: {default = 1e-8} Amount to increase denominator to avoid division by zero     
        """

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learning_rate * self.bias_corrected_weight_momentums[i] / (np.sqrt(self.bias_corrected_weight_cache[i]) + epsilon)

    
    
    def update_bias(self, learning_rate, epsilon):
        """
        returns updated bias values of neural net

        INPUT:
        learning_rate: learning rate to multiply delta by
        epsilon: {default = 1e-8} amount to increase denominator to avoid division by zero
        """
        
        for i in range(len(self.bias)):
            self.bias[i] = self.bias[i] - learning_rate * self.bias_corrected_bias_momentums[i] / (np.sqrt(self.bias_corrected_bias_cache[i]) + epsilon)
    

    
    def update_momentum(self, deceleration, weight_gradients, bias_gradients):
        """
        Updates current weight and bias momentum (biased)
        
        """
        
        for i in range(len(weight_gradients)):
            decelerated_previous_weight_momentum  = np.dot(deceleration, self.weight_momentum[i])
            weighted_average = 1 - deceleration
            weighted_weight_gradient = np.dot(weighted_average, weight_gradients[i])
            weight_momentum_update = decelerated_previous_weight_momentum + weighted_weight_gradient
            self.weight_momentum[i] = weight_momentum_update
            
            # calculate and append new bias momentum for layer i to array b
            decelerated_previous_bias_momentum = np.dot(deceleration, self.bias_momentum[i])
            weighted_bias_gradient = np.dot(weighted_average, bias_gradients[i])
            bias_momentum_update = decelerated_previous_bias_momentum + weighted_bias_gradient
            self.bias_momentum[i] = bias_momentum_update
#             self.bias_momentum[i] = ((deceleration * self.bias_momentum[i]) + ((1 - deceleration) * bias_gradients[i]))
            
        
    
    def bias_correct_momentum(self, t, deceleration):
        """
        Updates weight and bias momentum corrected for bias
        
        INPUT:
        beta: {default 0.9}
        t: epoch (start at 1. epoch(0) = 1)
        """
        
        correction = 1 - deceleration**t
        if self.store_metrics:
            self.all_momentum_corrections.append(correction)
        
        for i in range(len(self.weight_momentum)):
            self.bias_corrected_weight_momentums[i] = self.weight_momentum[i] / correction
            self.bias_corrected_bias_momentums[i] = self.bias_momentum[i] / correction
            
    
    def update_cache(self, decay, weight_gradients, bias_gradients):
        """
        Updates current weight and bias cache (biased)
        """
        
        for g in range(len(weight_gradients)):
            self.weight_cache[g] = (decay * self.weight_cache[g] + (1 - decay) * (weight_gradients[g] * weight_gradients[g]))
            self.bias_cache[g] = (decay * self.bias_cache[g] + (1 - decay) * (bias_gradients[g] * bias_gradients[g]))
            
    
    def bias_correct_cache(self, t, decay):
        """
        Updates weight and bias cache corrected for bias
        """
        
        correction = 1 - decay**t
        if self.store_metrics:
            self.all_cache_corrections.append(correction)
        
        for i in range(len(self.weight_cache)):
            self.bias_corrected_weight_cache[i] = self.weight_cache[i] / correction
            self.bias_corrected_bias_cache[i] = self.bias_cache[i] / correction

    
    
    def train(self, Xtrain, Ytrain, Xtest, Ytest, epochs, learning_rate, activation, regularization, gradient_descent = 'full', samples_per_batch = 100, deceleration = 0.9, epsilon = 1e-08, decay = 0.999):
        """
        Input:
        Xtrain: feature training data
        Ytrain: target training data
        Xtest: feature test data
        Ytest: target test data
        epochs: {int} number of training iterations 
        learning_rate: {float} 
        activation: {string} {'sigmoid','relu','tanh'} activation algorithm to be used on each hidden layer node
        regularization: {float} 
        gradient_descent: {string} {'full','stochastic','batch'} {default = 'full'} type of gradient descent used
        samples_per_batch: {int} {only if gradient_descent = 'batch'} {default = 100} number of samples to be used per batch of batch gradient descent
        deceleration: {float} {default = 0.9} used to calculate amount of previous gradient to be used in updated gradient (momentum)
        epsilon: {float} {default = 1e-8} small number added to the cache to avoid dividing by zero in weight and bias update algorithm
        decay: {float} {default = 0.999} number used to multiply cache by to keep it from growing too quickly (RMS-Prop). Also used to correct for bias in momentum and cache
        
        Output:
        Prints train cost, test cost, and error rate per 10 epochs
        Prints final score (accuracy) and error rate of test data
        Displays loss/cost over epochs for both training and testing datasets  
        """
        # set activation
        self.activation = activation
        
        # turn Ytrain and Ytest categorical data sets into one hot encoded data sets
        Ttrain = y_to_indicator(Ytrain)
        Ttest = y_to_indicator(Ytest)
        
        # set Xtrain number of features/categories = D
        D = Xtrain.shape[1]
        # set Ytrain number of categories/outcomes = K
        K = len(set(Ytrain))
        # set t = 1 for bias correction function
        t = 1

        # calculate and append w1 and b1 to arrays self.weights and self.bias
        self.weights.append(np.random.randn(D, self.hidden_nodes[0]) / np.sqrt(D))
        self.bias.append(np.random.randn(self.hidden_nodes[0]))

        # set first layer of weight momentums and bias corrected weight momentums to 0
        self.weight_momentum.append(np.zeros((D, self.hidden_nodes[0])))
        self.bias_corrected_weight_momentums.append(np.zeros((D, self.hidden_nodes[0])))
         
        # set first layer of weight_cache and bias corrected weight cache to 0
        self.weight_cache.append(np.zeros((D, self.hidden_nodes[0])))
        self.bias_corrected_weight_cache.append(np.zeros((D, self.hidden_nodes[0])))
        
        # set first layer of bias_momentum and bias corrected bias momentum to 0
        self.bias_momentum.append(np.zeros(self.hidden_nodes[0]))
        self.bias_corrected_bias_momentums.append(np.zeros(self.hidden_nodes[0]))
          
        # set first layer of bias_cache and bias corrected bias cache to 0
        self.bias_cache.append(np.zeros(self.hidden_nodes[0]))
        self.bias_corrected_bias_cache.append(np.zeros(self.hidden_nodes[0]))

        # set shapes and values for all hidden layer arrays
        for w in range(self.hidden_layers - 1):
            weights = np.random.randn(self.hidden_nodes[w], self.hidden_nodes[w+1])
            self.weights.append(weights)
            bias = np.random.randn(self.hidden_nodes[w+1])
            self.bias.append(bias)
            self.weight_momentum.append(np.zeros((self.hidden_nodes[w], self.hidden_nodes[w+1])))
            self.bias_corrected_weight_momentums.append(np.zeros((self.hidden_nodes[w], self.hidden_nodes[w+1])))
            self.bias_momentum.append(np.zeros(self.hidden_nodes[w+1]))
            self.bias_corrected_bias_momentums.append(np.zeros(self.hidden_nodes[w+1]))
            self.weight_cache.append(np.zeros((self.hidden_nodes[w], self.hidden_nodes[w+1])))
            self.bias_corrected_weight_cache.append(np.zeros((self.hidden_nodes[w], self.hidden_nodes[w+1])))
            self.bias_cache.append(np.zeros(self.hidden_nodes[w+1]))
            self.bias_corrected_bias_cache.append(np.zeros(self.hidden_nodes[w+1]))

        # set shapes and values for final layer arrays
        self.weights.append(np.random.randn(self.hidden_nodes[-1], K) / np.sqrt(self.hidden_nodes[-1]))
        self.bias.append(np.random.randn(K))
        self.weight_momentum.append(np.zeros((self.hidden_nodes[-1], K)))
        self.bias_corrected_weight_momentums.append(np.zeros((self.hidden_nodes[-1], K)))
        self.bias_momentum.append(np.zeros(K))
        self.bias_corrected_bias_momentums.append(np.zeros(K))
        self.weight_cache.append(np.zeros((self.hidden_nodes[-1], K)))
        self.bias_corrected_weight_cache.append(np.zeros((self.hidden_nodes[-1], K)))
        self.bias_cache.append(np.zeros(K))
        self.bias_corrected_bias_cache.append(np.zeros(K))

        if self.store_metrics:
            # append current weight momentums to all_weight_momentums array
            self.all_weight_momentums.append(self.weight_momentum)
            # append current weight cache to all_weight_caches array
            self.all_weight_caches.append(self.weight_cache)
            # append current bias momentum to all_bias_momentums array
            self.all_bias_momentums.append(self.bias_momentum)
            # append current bias cache to all_bias_caches array
            self.all_bias_caches.append(self.bias_cache)

        for e in range(epochs):
            
            ## NEEDS TO BE COMPLETED
            if gradient_descent == 'full':
            
                Z_train = [] # create array to store hidden layer values of train set
                Z_train.append(self.forward(Xtrain, self.weights[0], self.bias[0])) # calculate first hidden layer values by using Xtrain as input
                
        
                # go through calculating the rest of the hidden layers
                for i in range(self.hidden_layers - 1):
                    Z_train.append(self.forward(Z_train[i], self.weights[i+1], self.bias[i+1]))
                
                # append Z_train to all_hidden_layers array
                self.all_hidden_layers.append(Z_train)

                # calculate output
                Y_hat = self.softmax(Z_train[-1].dot(self.weights[-1]) + self.bias[-1])
                self.Yhats.append(Y_hat)

                # calculate train loss and append to train costs array
                Ctrain = self.cross_entropy2(Ytrain, Y_hat)
                self.train_costs.append(Ctrain)

                # create array to store hidden layer values of test set
                Z_test = []
                
                # calculate first hidden layer values by using Xtest as input
                Z_test.append(self.forward(Xtest, self.weights[0], self.bias[0]))

                # go through calculating the rest of the hidden layers (Z's) for the test data
                for i in range(self.hidden_layers - 1):
                    Z_test.append(self.forward(Z_test[i], self.weights[i+1], self.bias[i+1]))

                # calculate output of test data
                test_Y_hat = self.softmax(Z_test[-1].dot(self.weights[-1]) + self.bias[-1])
                
                # calculate the loss of the test data and append to test_costs array
                Ctest = self.cross_entropy2(Ytest, test_Y_hat)
                self.test_costs.append(Ctest)

                # print out cost and classification rate every 10 epochs
                if e%10 == 0:
                    error = error_rate(Ytest, self.predict(Xtest))
                    print(f'epoch:{e} train cost: {Ctrain} test cost:{Ctest} error rate:{error}')
                    print('')

                # calculate derivative deltas between layer gradients
                deltas = self.delta(T=Ttrain, Y_hat=Y_hat, Z=Z_train)
                
                # calculate gradients (deltas = bias gradients)
                weight_gradients = self.calculate_weight_gradients(D=deltas, Z=Z_train, X=Xtrain)
                bias_gradients = []
                for i in range(len(deltas)):
                    bias_gradients.append(np.sum(deltas[i], axis=0))
                
                # apply regularization if regularization > 0
                if regularization > 0.0:
                    for i in range(len(weight_gradients)):
                        weight_gradients[i] += (regularization * self.weights[i])
                        bias_gradients[i] += (regularization * self.bias[i])
                    
                    
                
                # update momentum
                self.weight_momentum, self.bias_momentum = self.update_momentum(deceleration, weight_gradients, bias_gradients)
                
                # update cache
                self.weight_cache, self.bias_cache = self.update_cache(decay, weight_gradients, bias_gradients)
                
                # bias corrections
                self.bias_correct_momentum(t = e + 1, deceleration = deceleration)
                self.bias_corrected_cache(t = e + 1, decay = decay)
                
                # update weights
                self.weights = self.update_weights(learning_rate=learning_rate, epsilon=epsilon)
                self.bias = self.update_bias(learning_rate=learning_rate, epsilon=epsilon)
                
                
                
                
                
            ## STOCHASTIC NEEDS TO BE COMPLETED    
            if gradient_descent == 'stochastic':
                
                Y_hats = [] # create array to store Y_hats of each batch
                Y_hats = np.array(Y_hats)
                
                # loop through each sample in Xtrain individually, calculating its gradient and updating the weights and bias
                for n in range(len(Xtrain)):
                    # print value of 'n' for debugging
#                     print(f'n = {n}')
                    
                    # print shape of Xtrain[n] for debugging
#                     print(f'Xtrain{n} shape: {Xtrain[n].shape}')
                    
                    # print reshape of Xtrain[n] for debugging to make sure its 1XD array
#                     print(f'Xtrain{n} reshape: {np.expand_dims(Xtrain[n], axis=1).T.shape}')
                    
                    Z_train = [] # create array to store hidden layer values of train set
                    
                    # print shapes of self.weights[0] and self.bias[0]
#                     print(f'shape self.weights[0]: {self.weights[0].shape}')
#                     print(f'shape self.bias[0]: {self.bias[0].shape}')
                    
                    Z_train.append(self.forward(np.expand_dims(Xtrain[n], axis=1).T, self.weights[0], self.bias[0])) # calculate first hidden layer values by using Xtrain as input
                    
                    # go through calculating the rest of the hidden layers
                    for i in range(self.hidden_layers - 1):
                        Z_train.append(self.forward(Z_train[i], self.weights[i+1], self.bias[i+1]))
                        
                    # print shape of Z_train for debugging
#                     print(f'Z_train shape: {np.array(Z_train).shape}')
                        
                    # calculate output
                    Y_hat = self.softmax(Z_train[-1].dot(self.weights[-1]) + self.bias[-1])
                    
                    # print shape of Y_hat array for debuggin
#                     print(f'Y_hat shape: {Y_hat.shape}')
                    
                    Y_hats = np.append(Y_hats, Y_hat)
                    
                    # output shape of Y_hats array for debugging (should be current n x k)
#                     print(f'Y_hats shape: {Y_hats.shape}')
                    
                    # print Ttrain shape for debugging
#                     print(f'Ttrain{n} shape: {Ttrain.shape}')
                    
                    deltas = self.delta(T=Ttrain[n], Y_hat=Y_hat, Z=Z_train)
                    
                    # print shape of deltas for debugging
#                     print(f'deltas shape: {np.array(deltas).shape}')

                    gradients = self.calculate_weight_gradients(D=deltas, Z=Z_train, X=Xtrain[n])
                    
                    # print shape of gradients for debugging
#                     print(f'gradients shape: {np.array(gradients).shape}')

                    self.weights = self.update_weights(G=gradients, learning_rate=learning_rate, regularization=regularization, deceleration=deceleration)
                    self.bias = self.update_bias(D=deltas, learning_rate=learning_rate, regularization=regularization, deceleration=deceleration)
                    
#                   # update weight velocities and bias velocities
                    self.weight_momentum, self.bias_momentum = self.update_momentum(deceleration, learning_rate, gradients, deltas)
                
                # reshape Y_hats to be an NxK array
                Y_hats = np.reshape(Y_hats, (len(Ytrain), K))
                
                # print reshaped Y_hats array for debugging
#                 print(f'reshaped Y_hats: {Y_hats.shape}')
                
                Ctrain = self.cross_entropy2(Ytrain, Y_hats)
                self.train_costs.append(Ctrain)
                
                Z_test = [] # create array to store hidden layer values of test set
                Z_test.append(self.forward(Xtest, self.weights[0], self.bias[0])) # calculate  first hidden layer values by using Xtest as input

                # go through calculating the rest of the hidden layers
                for i in range(self.hidden_layers - 1):
                    Z_test.append(self.forward(Z_test[i], self.weights[i+1], self.bias[i+1]))


                # calculate output
                test_Y_hat = self.softmax(Z_test[-1].dot(self.weights[-1]) + self.bias[-1])
                Ctest = self.cross_entropy2(Ytest, test_Y_hat)
                self.test_costs.append(Ctest)
                
                # print out cost and classification rate every 10 epochs
                if e%10 == 0:
                    error = error_rate(Ytest, self.predict(Xtest))
                    print(f'epoch:{e} train cost: {Ctrain} test cost:{Ctest} error rate:{error}')
                    print('')
                    
                    
                    
            if gradient_descent == 'batch':
                
                # create array to store Y_hats of each batch
                Y_hats = []
                Y_hats = np.array(Y_hats)
                
                # calculate number of batches
                num_batches = int(np.ceil(len(Xtrain) / samples_per_batch))
                
                # shuffle the training sets
                Xtrain, Ytrain, Ttrain = shuffle(Xtrain, Ytrain, Ttrain)
                
                # across each batch, calculate gradients and update weights and bias
                for j in range(num_batches):
                    X_batch = Xtrain[j * samples_per_batch:(j+1) * samples_per_batch]
                    Y_batch = Ytrain[j * samples_per_batch:(j+1) * samples_per_batch]
                    T_batch = Ttrain[j * samples_per_batch:(j+1) * samples_per_batch]
                    
                    # create array to store hidden layer values of batch train set
                    Z_train = []
                    
                    # calculate first hidden layer values by using Xtrain as input
                    Z_train.append(self.forward(X_batch, self.weights[0], self.bias[0]))
                    
                    # go through calculating the rest of the hidden layers
                    for i in range(self.hidden_layers - 1):
                        Z_train.append(self.forward(Z_train[i], self.weights[i+1], self.bias[i+1]))
                
                    # calculate output
                    Y_hat = self.softmax(Z_train[-1].dot(self.weights[-1]) + self.bias[-1])
                    
                    # Y_hats array to be used to calculate cross-entropy of epoch
                    Y_hats = np.append(Y_hats, Y_hat)
                    
                    # calculate derivative deltas between layer gradients 
                    deltas = self.delta(T=T_batch, Y_hat=Y_hat, Z=Z_train)
                    
                    # calculate weight gradients for each layer
                    weight_gradients = self.calculate_weight_gradients(D=deltas, Z=Z_train, X=X_batch)
                    
                    # calculate bias gradients for each layer
                    bias_gradients = []
                    for i in range(len(deltas)):
                        bias_gradients.append(np.sum(deltas[i], axis=0))
                        
                    # apply regularization if regularization > 0
                    if regularization > 0.0:
                        for i in range(len(weight_gradients)):
                            weight_gradients[i] += (regularization * self.weights[i])
                            bias_gradients[i] += (regularization * self.bias[i])
                            
                    # update momentum
                    self.update_momentum(deceleration, weight_gradients, bias_gradients)
                   
                    # update cache
                    self.update_cache(decay, weight_gradients, bias_gradients)
                    
                    # bias corrections
                    self.bias_correct_momentum(t = t, deceleration = deceleration)
                    self.bias_correct_cache(t = t, decay = decay)
                        
                    # increase t by 1
                    t += 1

                    # update weights and bias
                    self.update_weights(learning_rate=learning_rate, epsilon=epsilon)
                    self.update_bias(learning_rate=learning_rate, epsilon=epsilon)
                    
                    # if store_metrics = True, store all variables
                    if self.store_metrics:
                        # append to all Xbatches array
                        self.all_Xbatches.append(X_batch)
                        # append to all Ybatches array
                        self.all_Ybatches.append(T_batch)
                        # store hidden layer calculations to all_hidden_layers array
                        self.all_hidden_layers.append(Z_train)
                        # store Y_hat to all Yhats array
                        self.Yhats.append(Y_hat)
                        # store current bias_gradients to all_bias_gradients array
                        self.all_bias_gradients.append(bias_gradients)
                        # store deltas to all_deltas array
                        self.all_deltas.append(deltas)
                        # store current weight_gradients to all_weight_gradients array
                        self.all_weight_gradients.append(weight_gradients)
                        # store current wieght_momentum to all_weight_momentums array
                        self.all_weight_momentums.append(self.weight_momentum)
                        # store current bias momentum to all_bias_momentums array
                        self.all_bias_momentums.append(self.bias_momentum)
                        # store current weight cache to all_weight_caches array
                        self.all_weight_caches.append(self.weight_cache)
                        # store current bias cache to all_bias_caches array
                        self.all_bias_caches.append(self.bias_cache)
                        # store current bias corrected momentum and bias corrected cache to all_ arrays
                        self.all_bias_corrected_weight_momentums.append(self.bias_corrected_weight_momentums)
                        self.all_bias_corrected_bias_momentums.append(self.bias_corrected_bias_momentums)
                        self.all_bias_corrected_weight_caches.append(self.bias_corrected_weight_cache)
                        self.all_bias_corrected_bias_caches.append(self.bias_corrected_bias_cache)
                        # store new weights and biases to all_ arrays
                        self.all_weights.append(self.weights)
                        self.all_biases.append(self.bias)
                    
                    # calculate loss per batch and print loss of test set
#                     if e%10 == 0:
#                         # create array to store hidden layer values for test batch
#                         z_test = []
#                         # calculate first hidden layer of test batch
#                         z_test.append(self.forward(Xtest, self.weights[0], self.bias[0]))
#                         # go through calculating the rest of the hidden layers
#                         for i in range(self.hidden_layers - 1):
#                             z_test.append(self.forward(z_test[i], self.weights[i+1], self.bias[i+1]))
                        
#                         # calculate output
#                         test_Y_hat = self.softmax(z_test[-1].dot(self.weights[-1]) + self.bias[-1])
                        
#                         # calculate test loss/cost
#                         ctest = self.cross_entropy2(Ytest, test_Y_hat)
#                         # print test cost
#                         print(f'Cost at iteration{e}, batch{j}: {ctest}')
                        
#                         # calculate and print test error
#                         error = error_rate(Ytest, self.predict(Xtest))
#                         print(f'Error Rate: {error}')
                
                # reshape Y_hats to be an NxK array
                Y_hats = np.reshape(Y_hats, (len(Ytrain), K))
                
                # calculate training loss/cost and append to train_costs array
                Ctrain = self.cross_entropy2(Ytrain, Y_hats)
                self.train_costs.append(Ctrain)
                
                # create array to store hidden layer values of test set
                Z_test = []
                
                # calculate first hidden layer of test dataset
                Z_test.append(self.forward(Xtest, self.weights[0], self.bias[0])) # calculate first hidden layer values by using Xtest as input

                # go through calculating the rest of the hidden layers for the test datset
                for i in range(self.hidden_layers - 1):
                    Z_test.append(self.forward(Z_test[i], self.weights[i+1], self.bias[i+1]))


                # calculate output of test datset
                test_Y_hat = self.softmax(Z_test[-1].dot(self.weights[-1]) + self.bias[-1])
                
                # calculate test loss/cost and append to test_costs array
                Ctest = self.cross_entropy2(Ytest, test_Y_hat)
                self.test_costs.append(Ctest)
                
                # print out cost and classification rate every 10 epochs
                if e%10 == 0:
                    error = error_rate(Ytest, self.predict(Xtest))
                    print(f'epoch:{e} train cost: {Ctrain} test cost:{Ctest} error rate:{error}')
                    print('')
                    
                    
        # display final score
        self.final_score = self.score(Xtest,Ytest)
        print(f'final score:{self.final_score}')
        
        # display final error rate
        self.error_rate = error_rate(Ytest, self.predict(Xtest))
        print(f'Final error Rate: {self.error_rate}')
        
        # display cost per epoch of training and testing sets
        self.display_cost_per_epoch(self.train_costs, self.test_costs)
            
       
    def cross_validation(model, X, Y, K, epochs, learning_rate, activation, regularization, gradient_descent = 'full', samples_per_batch = 100, deceleration = 0.9, epsilon = 1e-08, decay = 0.999):
        """
        Performs cross validation on a passed in model and prints mean score
        
        Input:
        model: model object used for training
        X: feature data
        Y: target data
        K: number of cross-validation folds
        epochs: {int} number of training iterations 
        learning_rate: {float} 
        activation: {string} {'sigmoid','relu','tanh'} activation algorithm to be used on each hidden layer node
        regularization: {float} 
        gradient_descent: {string} {'full','stochastic','batch'} {default = 'full'} type of gradient descent used
        samples_per_batch: {int} {only if gradient_descent = 'batch'} {default = 100} number of samples to be used per batch of batch gradient descent
        deceleration: {float} {default = 0.9} used to calculate amount of previous gradient to be used in updated gradient (momentum)
        epsilon: {float} {default = 1e-8} small number added to the cache to avoid dividing by zero in weight and bias update algorithm
        decay: {float} {default = 0.999} number used to multiply cache by to keep it from growing too quickly (RMS-Prop). Also used to correct for bias in momentum and cache
        
        Output:
        prints score (accuracy) of each cross-validation fold and the mean score over all folds
        """
        # shuffle X and Y datasets
        X, Y = shuffle(X, Y)
        
        # determine the size of each cross validation fold
        sz = int(np.ceil(len(Y) / K))
        
        # create array to store final score (accuracy) of each trained batch
        scores = []

        for k in range(K):
            print(f'k = {k}')
            Xtr = np.concatenate([X[:k*sz,:], X[(k*sz+sz):,:]])
            Ytr = np.concatenate([Y[:k*sz], Y[(k*sz+sz):]])
            Xte = X[k*sz:(k*sz+sz),:]
            Yte = Y[k*sz:(k*sz+sz)]

            model.train(Xtr, Ytr, Xte, Yte, epochs, learning_rate, activation, regularization, gradient_descent, samples_per_batch, deceleration, epsilon, decay)
            y_hat = indicator_to_y(Yte)
            score = model.score(Xtr, y_hat)
            print(f'score: {score}')
            scores.append(score)
            print('')
            
            print('Scores:')
            print(scores)
            print(f'mean score: {np.mean(scores)}, std: {np.std(scores)}')
            print('')
            mean_scores.append([np.mean(scores), np.std(scores)])

        print('mean scores:')
        print(mean_scores)
        print('')

                
    def display_cost_per_epoch(self, train_costs, test_costs):
        """
        displays a graph showing cost over epochs for both training and testing sets

        Input:
        train_costs = array of training costs per epoch
        test_costs = array of testing costs per epoch
        
        Output:
        Displays graph of training and testing costs over epochs
        """

        plt.plot(train_costs, label=f'{self.activation} Train')
        plt.plot(test_costs, label=f'{self.activation} Test')
        plt.legend()
        plt.title(label=f'Score:{self.final_score*100}%')
        plt.show()
        
    
    def predict(self, X, return_argmax = True):
        """
        predicts an output (categorical) given input (X)
        
        Input:
        X: dataset to make predictions on
        return_argmax: {default = True} returns the index of greatest probability and therefore the
            prediction.  Otherwise returns array of probabilities per category
        
        Returns:
        Numpy array of predicted categories
        """
        
        # create array to store hidden layer values
        Z = []
        
        # calculate first hidden layer values by using Xtrain as input
        Z.append(self.forward(X, self.weights[0], self.bias[0]))

        # go through calculating the rest of the hidden layers
        for i in range(self.hidden_layers - 1):
            Z.append(self.forward(Z[i], self.weights[i+1], self.bias[i+1]))

        # calculate output
        Y_hat = self.softmax(Z[-1].dot(self.weights[-1]) + self.bias[-1])
        
        if return_argmax:
            return np.argmax(Y_hat, axis=1)
        else:
            return Y_hat
    
    def score(self, X,Y):
        """
        Returns an accuracy score given X and Y
        
        Input:
        X: feature dataset
        Y: target dataset
        
        Returns:
        Accuracy score of model on dataset
        """
        
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)

def comparator(models, Xtrain, Ytrain, Xtest, Ytest, epochs, training_iterations):
    """
    Compares multiple models to determine best hyperparameters
    
    Inputs:
    models: {list of dicts} list of models stored as dictionary setup as {'name':'string', 'hidden layers': int, 'hidden nodes per layer': list, 'learning rate': float, 'activation':'string', 'regularization': float, 'gradient descent': 'string', 'samples per batch': int}
    Xtrain: training feature data
    Ytrain: training target data
    Xtest: testing feature data
    Ytest: testing target data
    epochs: number of iterations per training session
    training_iterations: number of times to repeat training sessions (to calculate mean score of training sessions per model)
    
    Output:
    """
    costs = np.zeros([len(models),training_iterations])
    for i in range(len(models)):
        for k in range(training_iterations):
            models[i]['name'] = Logistic_Regressor(models[i]['hidden layers'], models[i]['hidden nodes per layer'])
            models[i]['name'].train(Xtrain, Ytrain, Xtest, Ytest, epochs, models[i]['learning rate'], models[i]['activation'], models[i]['regularization'], models[i]['gradient descent'], models[i]['samples per batch'])
            
            # add model score to correct column of scores array
            costs[i,k] = models[i]['name'].test_costs[-1]
            
    # print out final scores
    print('Final Costs')
    print(costs)
    
    # determine model with lowest cost
    lowest_cost_index = np.unravel_index(np.argmin(costs, axis=None), costs.shape)
    print(f'lowest cost: {costs[lowest_cost_index]}')
    print(f'Best model: {models[lowest_cost_index.shape[0]]["name"]}')
    
    for i in range(len(models)):
#         plt.plot(models[i]['name'].train_costs, label=f'{models[i]["hidden nodes per layer"]} {models[i]["activation"]} {models[i]["learning rate"]} Train')
        plt.plot(models[i]['name'].test_costs, label=f'{models[i]["hidden nodes per layer"]} {models[i]["activation"]} {models[i]["learning rate"]} Test {np.round(models[i]["name"].final_score,2)}%')
    
    plt.legend()
    plt.title(label='Model Comparator')
    plt.show()
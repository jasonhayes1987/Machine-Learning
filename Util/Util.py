from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

def getData(balance_ones=True, Ntest=1000):
    # images are 48x48 = 2304 size vectors
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    # shuffle and split
    X, Y = shuffle(X, Y)
    Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
    Xtest, Ytest = X[-Ntest:], Y[-Ntest:]


    return Xtrain, Ytrain, Xtest, Ytest

def class_balancer(X, Y, target_class):
    """
    takes in an X and Y dataset and returns it with the 'target_class' balanced
    
    INPUT:
    X: inputs
    Y: target outputs
    target_class: class to be balanced
    
    RETURNS:
    X, Y
    """
    
    # divide number of samples by number of classes to get mean
    mean = len(Y) / len(set(Y))

    # divide mean by number of samples in class to determine how many times larger mean is and round to nearest number
    times_larger = int(round(mean / len(Y[Y==target_class]),0))

    # # repeat class by 'times larger' variable to balance
    X0, Y0 = X[Y!=target_class, :], Y[Y!=target_class]
    X1 = X[Y==target_class, :]
    X1 = np.repeat(X1, times_larger, axis=0)
    X = np.vstack([X0, X1])
    Y = np.concatenate((Y0, [target_class]*len(X1)))
    
    return X, Y

def y_to_indicator(Y):
    # turn Y into an indicator matrix for training
    T = np.zeros((len(Y), len(set(Y))))
    for i in range(len(Y)):
        T[i, Y[i]] = 1

    return T

def indicator_to_y(T):
    # turn indicator (one hot encoded) matrix 'T' back into Y matrix of categories
    # ie [0,0,1] -> 2
    #    [1,0,0] -> 0
    
    return np.argmax(T, axis=1)

def generate_clusters(N,k,r):
    """
    returns 3 generated datasets of N elements.  The first is X, which is of shape NxD and divided into k clusters.
    The second is Y, which is of shape Nx1 and is the labels of the cluster of which X belongs.
    The third is a dataset of the centroids of each cluster
    
    N = number of data points
    k = number of clusters
    r = radius of clusters
    
    returns X,Y,C
    """
    points_per_cluster = int(N/k)
    
    X = []
    Y = []
    C = []
    for i in range(k):
        c = generate_centroid()
        C.append(c)
        for p in range(points_per_cluster):
            X.append(get_coordinates(c,np.random.rand()*r,np.random.randint(360)))
            Y.append(i)
    X = np.array(X)
    Y = np.array(Y)
    C = np.array(C)
    
    return X,Y,C

def get_coordinates(c,r,d):
    """
    c = centroid in (x,y) form
    r = radius
    d = degree angle from centroid to point
    
    returns: x,y coordinates
    
    """
    
    x = (np.cos(d)*r) + c[0]
    y = (np.sin(d)*r) + c[1]
    
    return x,y

def generate_centroid(Xrange=10, Yrange=10):
    """
    generates and returns a random coordinate from range parameter (Xrange, Yrange)
    
    returns array [x,y]
    """
    
    x = Xrange * np.random.rand()
    y = Yrange * np.random.rand()
    
    return [x,y]

def plot_labeled_data(x,y,c):
    """
    displays scatter plot of x, labeled with y, and centroids
    """

    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.scatter(c[:,0],c[:,1], c='red',marker='*',s=100)
    
def train_test_split(X, Y, percent_test_data = 20):
    
    X, Y = shuffle(X, Y)
    
    perc = percent_test_data / 100
    index_split = int(round(len(Y) * perc, ndigits=0))
    
    Xtrain = X[:-index_split]
    Ytrain = Y[:-index_split]
    Xtest = X[-index_split:]
    Ytest = Y[-index_split:]
    
    return np.array(Xtrain), np.array(Ytrain), np.array(Xtest), np.array(Ytest)

def error_rate(targets, predictions):
    return np.mean(targets != predictions)

def elu(Z, alpha):
    return Z if Z > 0 else alpha*(np.exp(Z)-1)
# vectorize function to be used on array
elu_vectorized = np.vectorize(elu)

def elu_prime(Z, alpha):
    return 1 if Z > 0 else Z + alpha
# vectorize function to be used on array
elu_prime_vectorized = np.vectorize(elu_prime)

def display_cost_per_epoch(train_costs, test_costs):
    """
    displays a graph showing cost over epochs for both training and testing sets

    Input:
    train_costs = array of training costs per epoch
    test_costs = array of testing costs per epoch

    Output:
    Displays graph of training and testing costs over epochs
    """

    plt.plot(train_costs, label='Train')
    plt.plot(test_costs, label='Test')
    plt.legend()
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title(label='Cost/Iteration')
    plt.show()
    
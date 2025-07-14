
import numpy as np

class LinearRegression:

    def __init__(self,learning_rate:float = 0.1):

        self.learning_rate = learning_rate
        self.W = None
        self.b = None
        self.y = None

    def initialize_parameters(self,n_features:int):


        self.W = np.random.randn(n_features)*0.1
        self.b = 0


    def forward_pass(self,X):

        predictions = np.dot(X,self.W) + self.b      
        return predictions

    def compute_cost(self, predictions):

        m = len(predictions)
        cost = pow((predictions - self.y),2).sum() / (2*m)

        return cost


    def backward_pass(self, predictions):

        m = len(predictions)
        errors = predictions - self.y
        self.dW = np.dot(errors,self.X) / m
        self.db = (predictions - self.y ) / m


    def learn(self, X,y, iterations:int = 100):

        self.X = X
        self.y = y
        self.initialize_parameters(X.shape[1])
        costs = []

        for i in range(iterations):
            predictions = self.forward_pass(X=X)
            cost = self.compute_cost(predictions=predictions)
            self.backward_pass(predictions)
            self.W -= self.learning_rate * self.dW
            self.b -= self.learning_rate * self.db
            costs.append(cost)

        print(costs)
            

        
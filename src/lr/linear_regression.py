
import numpy as np

class LinearRegression:

    def __init__(self,learning_rate:float = 0.1, max_iterations:int = 1000, tolerance = 1e-6):
        
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.W = None
        self.b = None

    def _initialize_parameters(self, n_features:int):

        self.W = np.random.randn(n_features)*0.1
        self.b = 0.0


    def compute_loss(self, y_true, predictions):

        m = len(predictions)
        loss = ((predictions - y_true) ** 2).sum() / m

        return loss
    

    def compute_gradients(self,X,y_true,predictions):

        residuals = predictions - y_true
        m = len(residuals)

        dW = np.dot(X.T,residuals) / m
        db = np.sum(residuals) / m

        return dW, db


    def forward_pass(self,X):

        predictions = np.dot(X,self.W) + self.b

        return predictions
    
    def backward_pass(self,dW,db):

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db


    def fit(self,X,y):

        losses = []
        for i in range(self.max_iterations):
            predictions = self.forward_pass(X)
            loss = self.compute_loss(y,predictions)
            dW, db = self.compute_gradients(X,y,predictions)
            self.backward_pass(dW,db)
            losses.append(loss)

            if i > 0 and abs(losses[-2] - losses[-1]) < self.tolerance:
                print(f"Converged at iteration {i}")
                break

            
        print(losses)

        
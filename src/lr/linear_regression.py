
import numpy as np

class LinearRegression:

    def __init__(self,
                 learning_rate:float = 0.1,
                   max_iterations:int = 1000,
                     tolerance = 1e-6,
                     regularization:str = "none",
                     alpha:float = 0.0):
        
    
        assert regularization in {"none", "l1", "l2"}, "Invalid regularization type"

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.W = None
        self.b = None
        self.regularization = regularization
        self.alpha = alpha

    def _initialize_parameters(self, n_features:int):

        self.W = np.random.randn(n_features)*0.1
        self.b = 0.0


    def _compute_loss(self, y_true, predictions):



        m = len(predictions)
        loss = ((predictions - y_true) ** 2).sum() / m

        if self.regularization =="l1":
            reg = self.alpha * np.abs(self.W).sum()
        elif self.regularization =="l2":
            reg = self.alpha * np.sum(self.W ** 2)
        else:
            reg = 0.0

        return loss + reg
    

    def _compute_gradients(self,X,y_true,predictions):

        residuals = predictions - y_true
        m = len(residuals)

        dW = 2 * np.dot(X.T,residuals) / m
        db = 2 * np.sum(residuals) / m

        if self.regularization == "l1":
            dW += self.alpha * np.sign(self.W) ## izvod od alpha*|W| je +/- alpha

        elif self.regularization =="l2":
            dW += 2 * self.alpha * self.W  ## izvod od alpha*W^2 je 2 alpha w

        return dW, db


    def _forward_pass(self,X):

        predictions = np.dot(X,self.W) + self.b

        return predictions
    
    def _backward_pass(self,dW,db):

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

    def fit(self,X,y):

        n_features = X.shape[1]
        self._initialize_parameters(n_features=n_features)

        losses = []
        for i in range(self.max_iterations):
            predictions = self._forward_pass(X)
            loss = self._compute_loss(y,predictions)
            dW, db = self._compute_gradients(X,y,predictions)
            self._backward_pass(dW,db)
            losses.append(loss)

            if i > 0 and abs(losses[-2] - losses[-1]) < self.tolerance:
                print(f"Converged at iteration {i}")
                break

    def predict(self, X_test):
        return self._forward_pass(X_test)





        

import numpy as np
import pandas as pd


class LinearRegression:

    def __init__(self,
                 learning_rate:float = 0.1,
                   max_iterations:int = 1000,
                     tolerance = 1e-6,
                     regularization:str = "none",
                     regularization_rate:float = 0.0,
                     momentum:float =0.0,
                     decay_rate:float= 0.0 ):
        
    
        assert regularization in {"none", "l1", "l2"}, "Invalid regularization type"

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.W = None
        self.b = None
        self.regularization = regularization
        self.regularization_rate = regularization_rate
        self.momentum = momentum
        self.decay_rate = decay_rate
        self._momentum_w = None
        self._momentum_b = None
        self.iteration_step = 0
        
        
    

        

    def _initialize_parameters(self, n_features:int):

        self.W = np.random.randn(n_features)*0.1
        self.b = 0.0

        self._momentum_b = 0.0
        self._momentum_w = np.zeros(n_features)
        

    def _forward_pass(self,X):

        predictions = np.dot(X,self.W) + self.b
        return predictions


    def _compute_loss(self, y_true, predictions):

        m = len(predictions)
        loss = ((predictions - y_true) ** 2).sum() / m

        if self.regularization =="l1":
            reg = self.regularization_rate * np.abs(self.W).sum()
        elif self.regularization =="l2":
            reg = self.regularization_rate * np.sum(self.W ** 2)
        else:
            reg = 0.0

        return loss + reg
    

    def _compute_gradients(self,X,y_true,predictions):

        residuals = predictions - y_true
        m = len(residuals)

        dW = 2 * np.dot(X.T,residuals) / m
        db = 2 * np.sum(residuals) / m

        if self.regularization == "l1":
            dW += self.regularization_rate * np.sign(self.W) ## izvod od alpha*|W| je +/- alpha

        elif self.regularization =="l2":
            dW += 2 * self.regularization_rate * self.W  ## izvod od alpha*W^2 je 2 alpha w

        return dW, db

    
    def _backward_pass(self,dW,db):

        self.iteration_step += 1
        lr = self.learning_rate / (1 + self.decay_rate*self.iteration_step)


        if self.momentum > 0.0:

            self._momentum_w = self.momentum * self._momentum_w + lr*dW
            self._momentum_b = self.momentum * self._momentum_b + lr*db

            self.W = self.W - self._momentum_w
            self.b = self.b - self._momentum_b


        else:
            self.W = self.W - lr * dW
            self.b = self.b - lr* db



    def fit(self,X:pd.DataFrame, y:pd.Series):

        X = X.values
        y = y.values

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
        return losses
    


    def fit_sgd(self,X:pd.DataFrame ,y: pd.Series, batch_size:int = 1, epochs:int = 10):


        n_rows = X.shape[0]
        n_features = X.shape[1]
        self._initialize_parameters(n_features=n_features)
        losses = []
        
        for epoch in range(epochs):

            epoch_loss = 0.0
            shuffled_ids = np.random.permutation(n_rows)
            X_shuffled = X.iloc[shuffled_ids].values
            y_shuffled = y.iloc[shuffled_ids].values
        
        
            for start in range(0,n_rows,batch_size):

                
                x_batch = X_shuffled[start:start+batch_size]
                y_batch = y_shuffled[start:start+batch_size]

                predictions = self._forward_pass(x_batch)
                loss = self._compute_loss(y_batch,predictions)
                dW, db = self._compute_gradients(x_batch,y_batch,predictions)
                self._backward_pass(dW,db)

                epoch_loss+=loss * len(y_batch)

            avg_loss = epoch_loss / n_rows

            losses.append(avg_loss)
            
            if epoch > 0 and (losses[-2] - losses[-1])<self.tolerance:
                print(f"Converged at epoch {epoch}")
                break
            
        return losses


            
        

    def predict(self, X_test:pd.DataFrame):
        
        X_test = X_test.values 
        return self._forward_pass(X_test)





        
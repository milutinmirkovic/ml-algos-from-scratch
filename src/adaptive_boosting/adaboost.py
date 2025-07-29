import pandas as pd
import numpy as np
from typing import Optional, List,Type, Tuple
from sklearn.tree import DecisionTreeClassifier
import inspect

class Adaboost:

    def __init__(self, num_learners:int  = 10, learning_rate:float = 1, base_models: Optional[List[Type]] = None):

        self.num_learners = num_learners
        self.learning_rate = learning_rate
        
        if base_models is None:
            self.base_models = [DecisionTreeClassifier]
        else:
            self.base_models = base_models

        self.models = []  # lista modela
        self.model_weights = [] # tezina za svaki model


    def learn(self, X: pd.DataFrame, y: pd.Series):
        
        X = X.values
        y = y.values

        if  y.dtype == bool:
            y = y.astype(int)

        y = y * 2 - 1 ## y mora da bude -1 za false i 1 za true
        assert np.all(np.isin(y, [-1, 1])), "Izlazna promenljiva mora imati '1' za pozitivnu klasu i '-1' za negativnu klasu"

        n_rows = X.shape[0]
        sample_weights = np.full(shape=n_rows, fill_value= (1 / n_rows))

        for i in range(self.num_learners):

            learner_class = np.random.choice(self.base_models)
            learner = learner_class()
            signature = inspect.signature(learner.fit)
            if "sample_weight" in signature.parameters:
                learner.fit(X,y,sample_weights)
            else:
                raise ValueError(f"{learner_class.__name__} ne podrzava sample_weight")
            
            preds = learner.predict(X)
            errors = (preds!= y).astype(float)
            weighted_error = np.dot(errors, sample_weights) 

            if weighted_error == 0:
                weighted_error = 1e-6

            w = 0.5 * np.log((1 - weighted_error) / weighted_error)
            factor = np.exp(-self.learning_rate * w * y * preds)

            sample_weights = sample_weights * factor
            sample_weights = sample_weights / sample_weights.sum()

            self.models.append(learner)
            self.model_weights.append(w)

    def predict(self, X: pd.DataFrame) -> List[Tuple[int, float]]:
            """
            Makes predictions and returns the class and confidence score.
            """
            X_vals = X.values
            n_samples = X_vals.shape[0]
            ensemble_prediction_sum = np.zeros(n_samples)

            for model, w in zip(self.models, self.model_weights):
                prediction = model.predict(X_vals)
                ensemble_prediction_sum += self.learning_rate * w * prediction
                
            final_predictions = np.sign(ensemble_prediction_sum)
            
            total_weight_sum = np.sum(self.model_weights) * self.learning_rate
            confidence = np.clip(np.abs(ensemble_prediction_sum) / total_weight_sum, 0, 1)

            return list(zip(final_predictions.astype(int).tolist(), confidence.tolist()))
            












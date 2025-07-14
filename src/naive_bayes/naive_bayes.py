import pandas as pd
from core.features import get_features
from core.stats import mean, variance,gaussian_pdf
import math
import numpy as np



class NaiveBayes:

    def __init__(self,alpha:float = 0.01):
        self.alpha = alpha
        self.model = None

    def learn(self,X:pd.DataFrame, y:pd.Series):
        self.model = {}
        self.model['priors'] =(y.value_counts(normalize=True))
        self.features = get_features(X)

        for ft in self.features:
            if ft.is_continuous:
                stats = {}
                for cls in y.unique():
                    values = X[ft.name][y == cls].tolist()
                    stats[cls] = {
                        "mean": mean(values),
                        "variance": max(variance(values),1e-6)
                    }

                self.model[ft.name] = stats
            else:
                freqs = pd.crosstab(X[ft.name], y) 


                conditional_probs = (freqs + self.alpha) / ( freqs.sum(axis=0) + self.alpha*freqs.shape[0] )

                self.model[ft.name] = (conditional_probs)


        self.model["features"] = self.features
        
        return self.model
    
    def show_model(self):
        print("=== Priors ===")
        print(self.model["priors"])
        for ft in self.features:
            print(f"\n=== Feature: {ft.name} ===")
            print(self.model[ft.name])
    
    def predict(self, test_data:pd.DataFrame):

        results = []
        features = self.features
        classes = self.model['priors'].index

        for _,row in test_data.iterrows():
            log_probs = {}
            for cls in classes:
                log_prob = math.log(self.model['priors'][cls])
                #log_prob = self.model['priors'][cls]

                for ft in features:
                    val = row[ft.name]
                    if ft.is_continuous:
                        stats = self.model[ft.name][cls]
                        prob = gaussian_pdf(val,stats["mean"],stats["variance"])
                    else:
                        prob = self.model[ft.name].loc[val,cls]

                    log_prob += math.log(prob)
                    #log_prob *= prob
                log_probs[cls] = log_prob
            best_cls = max(log_probs,key=log_probs.get)
            results.append(best_cls)    
        return results        




        




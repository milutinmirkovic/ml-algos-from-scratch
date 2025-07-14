from lr.linear_regression import LinearRegression

import pandas as pd 
from core.stats import standarization

data = pd.read_csv(r"C:\ML from scratch\ml-algos-from-scratch\src\lr\boston.csv")
model = LinearRegression(learning_rate=0.25)



X = data.drop("MEDV",axis=1)
y = data["MEDV"]

X = (X - X.mean()) / X.std()



model.learn(X=X,y=y, iterations=5000)

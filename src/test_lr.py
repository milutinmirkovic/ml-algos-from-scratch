from lr.linear_regression import LinearRegression 
from core.metrics import mse, rmse, mae, r_squared

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLR




data = pd.read_csv(r"C:\ML from scratch\ml-algos-from-scratch\src\lr\boston.csv")
X = data.drop("MEDV", axis=1)
y = data["MEDV"]



X = (X - X.mean()) / X.std()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std



model = LinearRegression(
    learning_rate=0.1,
    max_iterations=1000,
    regularization="l1",
    regulatization_rate=0.3,
    momentum=0.1
)

model.fit(X_train,y_train)

preds = model.predict(X_test)

print(f"RMSE: {rmse(y_test,preds)}")
print(f"R squared: {r_squared(y_test,preds)}")



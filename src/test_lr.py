import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLR
from lr.linear_regression import LinearRegression
from core.metrics import mae, rmse, r_squared

# Load and split data
df = pd.read_csv(r"C:\ML from scratch\ml-algos-from-scratch\src\datasets\boston.csv")
X = df.drop(columns="MEDV")
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
X_mean, X_std = X_train.mean(), X_train.std()
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Define models
models = {
    "No Reg": LinearRegression(learning_rate=0.1, max_iterations=100,decay_rate=0.01),
    "SGD No Reg": LinearRegression(learning_rate=0.1, max_iterations=100,decay_rate=0.01),
    "L2 Reg": LinearRegression(learning_rate=0.01, max_iterations=100, regularization="l2", regularization_rate=0.1),
    "Sklearn": SklearnLR()
}

# Train and evaluate
for name, model in models.items():
    if name =="SGD No Reg":
        model.fit_sgd(X_train, y_train,epochs = 2)
    if name == "Sklearn":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} | MAE: {mae(y_test, y_pred):.4f} | RMSE: {rmse(y_test, y_pred):.4f} | R2: {r_squared(y_test, y_pred):.4f}")

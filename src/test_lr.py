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

# Convert to NumPy for custom model
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values

# ----------------------------
# 🛠 Your Implementation
# ----------------------------
lrs = [0.01,0.05,0.1]
alphas = [0.0,0.1,0.2,0.3]
regs = ["none","l1","l2"]
res = []

for lr in lrs:
    for alpha in alphas:
        for reg in regs:
            model = LinearRegression(learning_rate=lr, max_iterations=1000, regularization=reg,alpha=alpha)
            model.fit(X_train_np, y_train_np)
            y_pred_custom = model.predict(X_test_np)

            mse_val = mse(y_test_np, y_pred_custom)
            mae_val = mae(y_test_np, y_pred_custom)
            rmse_val = rmse(y_test_np, y_pred_custom)
            r_squared_val = r_squared(y_test_np, y_pred_custom)

            result = {
                "RMSE":rmse_val,
                "R squared":r_squared_val,
                "learning_rate":lr,
                "regularization":reg,
                "alpha":alpha,
                }
            res.append(result)
            
best = max(res, key=lambda r: r["R squared"])
print("🔍 Best result (by RMSE):")
print(best)  




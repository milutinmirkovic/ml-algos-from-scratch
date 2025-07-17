import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLR
from lr.linear_regression import LinearRegression
from core.metrics import mse, rmse, mae, r_squared

data = pd.read_csv(r"C:\ML from scratch\ml-algos-from-scratch\src\lr\boston.csv")
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_mean = X_train.mean()
X_std = X_train.std()
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

models = {
    "BGD (no reg)": LinearRegression(learning_rate=0.1, max_iterations=100,decay_rate = 0.01),
    "BGD":LinearRegression(learning_rate=0.01, max_iterations=100,decay_rate = 0.01,momentum=0.2),
    "BGD momentum":LinearRegression(learning_rate=0.01, max_iterations=100,decay_rate = 0.01),
    "BGD l1":LinearRegression(learning_rate=0.01, max_iterations=100,decay_rate = 0.01,regularization="l1",regularization_rate=0.1),
    "BGD l2":LinearRegression(learning_rate=0.01, max_iterations=100,decay_rate = 0.01,regularization="l2",regularization_rate=0.1),
    "BGD l2":LinearRegression(learning_rate=0.01, max_iterations=100,decay_rate = 0.01,regularization="l2",regularization_rate=0.1),


}

losses = {}
results = {}

for name, model in models.items():
    if "SGD" in name:
        loss = model.fit_sgd(X_train, y_train, batch_size=16, epochs=100)
    else:
        loss = model.fit_sgd(X_train, y_train,batch_size=2,epochs=4)
    y_pred = model.predict(X_test)
    losses[name] = loss
    results[name] = {
        "MAE": mae(y_test, y_pred),
        "RMSE": rmse(y_test, y_pred),
        "R2": r_squared(y_test, y_pred)
    }

sk_model = SklearnLR()
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)
results["Sklearn"] = {
    "MAE": mae(y_test, sk_preds),
    "RMSE": rmse(y_test, sk_preds),
    "R2": r_squared(y_test, sk_preds)
}

for name, metrics in results.items():
    print(f"{name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print()

plt.figure(figsize=(10, 5))
for name, l in losses.items():
    plt.plot(l, label=name)
plt.xlabel("Epoch / Iteration")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.tight_layout()
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from naive_bayes.naive_bayes import NaiveBayes
from core.metrics import confusion_matrix, display_confusion_matrix, classification_metrics

df = pd.read_csv(r"C:\ML from scratch\ml-algos-from-scratch\src\datasets\drug.csv")

X = df.drop(columns="Drug")
y = df["Drug"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = NaiveBayes(alpha=0.1)
model.learn(X_train, y_train)
predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)
display_confusion_matrix(cm)
print(classification_metrics(cm))
print("\nClassification report:\n", classification_report(y_test, predictions))

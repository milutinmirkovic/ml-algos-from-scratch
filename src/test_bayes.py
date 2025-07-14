import sys
import os
from sklearn.metrics import classification_report

# Add 'src' to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from naive_bayes.naive_bayes import NaiveBayes
from core.metrics import confusion_matrix, display_confusion_matrix,classification_metrics

# Full dataset
df = pd.read_csv(r"C:\ML from scratch\ml-algos-from-scratch\src\Ukradeni.csv")
X = df.drop(columns="Ukraden")
y = df["Ukraden"]


df_test = pd.read_csv(r"C:\ML from scratch\ml-algos-from-scratch\src\test_set.csv")
X_test = df_test.drop(columns="Drug")
y_test = df_test["Drug"]


model = NaiveBayes(alpha=0.1)
model.learn(X,y)
model.show_model()




# predictions = model.predict(X_test)
# print(predictions)

# conf_matrix = confusion_matrix(y_test,predictions)
# display_confusion_matrix(conf_matrix)

# metrics = classification_metrics(conf_matrix)
# print(metrics)



# print("===============report===========")
# print(classification_report(y_test,predictions))
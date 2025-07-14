import sys
import os
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

# Add 'src' to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from naive_bayes.naive_bayes import NaiveBayes
from core.metrics import confusion_matrix, display_confusion_matrix,classification_metrics

# Full dataset
df = pd.read_csv(r"C:\Users\milutin.mirkovic\ml-algos-from-scratch\src\nina_test\drug.csv")

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
X_train = train_df.drop(columns = "Drug")
X_test = test_df.drop(columns = "Drug")
y_train = train_df['Drug']
y_test = test_df['Drug']





model = NaiveBayes(alpha=0.1)
model.learn(X_train,y_train)
model.show_model()




predictions = model.predict(X_test)

conf_matrix = confusion_matrix(y_test,predictions)
display_confusion_matrix(conf_matrix)

metrics = classification_metrics(conf_matrix)
print(metrics)



# print("===============report===========")
# print(classification_report(y_test,predictions))
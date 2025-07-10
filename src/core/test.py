from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import random

from metrics import classification_metrics,display_confusion_matrix,confusion_matrix

# ---- Your functions go here (confusion_matrix, display_confusion_matrix, classification_metrics) ----
# Paste your full implementations above this line before running the test

# ------------------ BINARY CLASSIFICATION TEST ------------------
print("=== BINARY CLASSIFICATION ===")
y_true_bin = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
y_pred_bin = [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0]

# Sklearn metrics
print("\n--- sklearn ---")
print("Confusion Matrix:")
print(sk_confusion_matrix(y_true_bin, y_pred_bin))
print("Accuracy:", accuracy_score(y_true_bin, y_pred_bin))
print("Precision:", precision_score(y_true_bin, y_pred_bin, zero_division=0))
print("Recall:", recall_score(y_true_bin, y_pred_bin, zero_division=0))
print("F1 Score:", f1_score(y_true_bin, y_pred_bin, zero_division=0))

# Your metrics
print("\n--- Your implementation ---")
cm_bin = confusion_matrix(y_true_bin, y_pred_bin)
display_confusion_matrix(cm_bin)
metrics_bin = classification_metrics(cm_bin)
for k, v in metrics_bin.items():
    print(f"{k}: {v:.4f}")


# ------------------ MULTICLASS CLASSIFICATION TEST ------------------
print("\n\n=== MULTICLASS CLASSIFICATION ===")
y_true_multi = [random.choice([0, 1, 2]) for _ in range(30)]
y_pred_multi = [random.choice([0, 1, 2]) for _ in range(30)]

# Sklearn metrics
print("\n--- sklearn ---")
print("Confusion Matrix:")
print(sk_confusion_matrix(y_true_multi, y_pred_multi))
print("Accuracy:", accuracy_score(y_true_multi, y_pred_multi))
print("Precision (macro):", precision_score(y_true_multi, y_pred_multi, average='macro', zero_division=0))
print("Recall (macro):", recall_score(y_true_multi, y_pred_multi, average='macro', zero_division=0))
print("F1 Score (macro):", f1_score(y_true_multi, y_pred_multi, average='macro', zero_division=0))

# Your metrics
print("\n--- Your implementation ---")
cm_multi = confusion_matrix(y_true_multi, y_pred_multi)
display_confusion_matrix(cm_multi)
metrics_multi = classification_metrics(cm_multi)
for k, v in metrics_multi.items():
    print(f"{k}: {v:.4f}")

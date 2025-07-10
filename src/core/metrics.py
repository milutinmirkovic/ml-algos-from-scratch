from typing import List, Union, Dict
from math import sqrt


def confusion_matrix(
        y_true: List[Union[int,str]],
        y_pred: List[Union[int,str]]) -> Dict[Union[int,str],Dict[Union[int,str],int]]:      
    
    matrix = {}
    labels = set(y_true+y_pred)

    for true_label in labels:
        matrix[true_label] = {pred_label: 0 for pred_label in labels}

    for yt,yp in zip(y_true,y_pred):
        matrix[yt][yp] += 1

    return matrix

def display_confusion_matrix(
    matrix: Dict[Union[int, str], Dict[Union[int, str], int]]
) -> None:

    labels = sorted(matrix.keys())
    print("Confusion Matrix:")
    print("       Predicted")
    print("     " + "  ".join(f"{l:>4}" for l in labels))
    print("     " + "-" * (5 * len(labels)))

    for true_label in labels:
        row = [matrix[true_label][pred_label] for pred_label in labels]
        print(f"{true_label:>4} | " + "  ".join(f"{count:>4}" for count in row))


def classification_metrics(
    matrix: Dict[Union[int, str], Dict[Union[int, str], int]]
) -> Dict[str, float]:

    labels = sorted(matrix.keys())

    # Automatically treat as binary if 2x2 matrix
    if len(labels) == 2:
        pos_label = 1 if 1 in labels else labels[1]
        neg_label = [l for l in labels if l != pos_label][0]

        TP = matrix[pos_label][pos_label]
        FP = matrix[neg_label][pos_label]
        FN = matrix[pos_label][neg_label]
        TN = matrix[neg_label][neg_label]
        total = TP + FP + FN + TN

        accuracy = (TP + TN) / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F-1": f1
        }

    else:
        # Macro average for multiclass
        accuracies, recalls, precisions, f1s = [], [], [], []
        for label in labels:
            TP = matrix[label][label]
            FP = sum(matrix[other][label] for other in labels if other != label)
            FN = sum(matrix[label][other] for other in labels if other != label)
            total = sum(sum(row.values()) for row in matrix.values())
            TN = total - TP - FP - FN

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        

        total_correct = sum(matrix[label][label] for label in labels)
        total_samples = sum(sum(row.values()) for row in matrix.values())
        
        return {
            "Accuracy": total_correct / total_samples,
            "Precision": sum(precisions) / len(precisions),
            "Recall": sum(recalls) / len(recalls),
            "F-1": sum(f1s) / len(f1s)
        }
        
    
def mse(y_true,y_pred):
   errors = [pow((yt-yp),2) for yt,yp in zip(y_true,y_pred)]
   return sum(errors) / len(y_pred)

def rmse(y_true,y_pred):
    return sqrt(mse(y_true,y_pred))

def mae(y_true,y_pred):
    errors = [abs(yt - yp) for yt ,yp in zip(y_true,y_pred)]
    return sum(errors) / len(y_pred)
        
def r_squared(y_true,y_pred):
    y_mean = sum(y_true) / len(y_true)
    ss_tot = sum([pow(yt-y_mean,2) for yt in y_true])
    ss_res =sum([pow((yt-yp),2) for yt,yp in zip(y_true,y_pred)])

    return 1 - (ss_res / ss_tot)
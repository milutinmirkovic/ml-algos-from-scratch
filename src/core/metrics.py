from typing import List, Union, Dict


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


    
    
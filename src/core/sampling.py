import numpy as np
import pandas as pd

from typing import Tuple, Union, Optional

def split_dataset(data: pd.DataFrame | pd.Series,
                  train_size: float = 0.7,
                  valid_size: float | None = None,
                  random_state: int = 42,
                  shuffle: bool = True) -> Union[
                    Tuple[pd.DataFrame,pd.DataFrame],
                    Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]  
                  ]:
    
    np.random.seed(random_state)
    indices = data.index.to_numpy()

    if shuffle:
        np.random.shuffle(indices)
    
    n_total = len(data)
    n_train = int(train_size*n_total)

    if valid_size is None:
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        return data.loc[train_idx],data.loc[test_idx]
    
    n_val = int(valid_size * n_total)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train + n_val:]

    return data.loc[train_idx], data.loc[valid_idx], data.loc[test_idx]


def bootstrap(
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        return_out_of_bag: bool = False,
        random_state: int = 42)-> Union[
            Tuple[np.ndarray,np.ndarray],
            Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]
            ]:
    
    rng = np.random.default_rng(random_state)
    n = len(X)
    indices = rng.integers(low = 0, high=n, size=n)
    oob_indices = np.setdiff1d(np.arange(n),indices)

    if isinstance(X,(pd.Series, pd.DataFrame)):
        X_sample = X.iloc[indices]
        X_oob = X.iloc[oob_indices] if return_out_of_bag else None
    else:
        X_sample = X[indices]
        X_oob = X[oob_indices] if return_out_of_bag else None

    if y is not None:
        if isinstance(y,pd.Series):
            y_sample = y.iloc[indices]
            y_oob = y.iloc[oob_indices] if return_out_of_bag else None
        else:
            y_sample = y[indices]
            y_oob = y[y_oob] if return_out_of_bag else None
    else:
        y_sample = y
        y_oob = None
    if return_out_of_bag:
        return X_sample, y_sample, X_oob, y_oob
    return X_sample,y_sample

        

    

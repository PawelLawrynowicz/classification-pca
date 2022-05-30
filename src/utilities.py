import pandas as pd

def one_hot(dataset: pd.DataFrame, column: str):
    replaced_feature = dataset.pop(column)
    dummies = pd.get_dummies(replaced_feature)
    return dataset.join(dummies)
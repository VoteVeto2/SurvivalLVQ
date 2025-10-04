import pandas as pd
import numpy as np
from SurvSet.data import SurvLoader

def list_datasets():
    loader = SurvLoader()
    return loader.df_ds.ds.to_numpy()

def load_dataset(name):
    loader = SurvLoader()
    df, ref = loader.load_dataset(ds_name=name).values()

    # one-hot encode the categorical variables
    one_hot = df.filter(regex='^fac', axis=1)
    if not one_hot.empty:
        one_hot = pd.get_dummies(one_hot, drop_first=True)

    numerical = df.filter(regex='^num', axis=1)
    time = df.time
    event = df.event
    data = pd.concat([one_hot, numerical], axis=1)

    fac_col_ids = np.arange(0, one_hot.shape[1])
    num_col_ids = np.arange(one_hot.shape[1], one_hot.shape[1] + numerical.shape[1])
    return data, fac_col_ids, num_col_ids, time, event

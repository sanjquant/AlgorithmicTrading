
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin


# define seed
def set_seeds(seed=42): 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

## Create a custom day transformer :
class DayTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.data =pd.Data




# create function to read locally stored file : 
def getData(filename: str):
    path = f'../data/{filename}.csv'
    df = pd.read_csv(path)
    df.datetime = pd.to_datetime(df.datetime)
    df = (df.set_index('datetime', drop=True).drop('symbol', drop= True))
    df['day'] = df.index.day_name()

    # add day parts
    df['hours'] = df.index.hour
    df['hours'] = df['hours'].apply(day_parts)
    return df


# create function to group trade hours :
def day_parts(hour):
    if hour in [9,10, 11]:
        return "morning"
    elif hour in [12, 13]:
        return "noon"
    elif hour in [14, 15, 16, 17, 18, 19]:
        return " afternoon"


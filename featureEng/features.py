from dataclasses import dataclass
from src.config import *
from src.helper import *

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 1000)

@dataclass
class CrateFeatures(object):

    ## feature Name : 
    open_2_close = 'o2c'
    high_2_low = 'h2l'
    open ='open'
    close = 'close'
    high = 'high'
    low = 'low'
    hours = 'hours'

    """
    Feature Specification
    Features are also known as an independent variable which are used to determine the value of the target variable. 
    We will add the absolute change between open-close price and high-low price as predictors along with volume, date-time features.
    
    """

    def create_features(self, dataframeObject):
        df = dataframeObject.copy()
        df[self.open_2_close] = df[self.open]-df[self.close]
        df[self.high_2_low] = df[self.high]-df[self.low]

        df_updated = df.dropna([self.open, self.high, self.low, self.close], axis=1)

        # rearrange columns
        cols = df_updated.columns; cols =list(cols); cols = cols[-2:]+cols[:-2]
        cols
        df_updated = df_updated[cols]

        return df_updated


    """
    Label Definition
    Label or the target variable is also known as the dependent variable. Here, 
    the target variable is whether Nifty50 Index's return will close up or down on the next bar. If the next bar return is greater than zero, then we will initiate a buy on the index, else no new position is made.
    We assign a value of +1 for the buy signal and 0 otherwise to target variable. The target can be described as :
    y_t =  1 if r_t+1 >0
        0, if r_t+1 , otherwise
        where , r_t+1 is the 1-bar forward return 

    """
    def get_returns(self, dataframeObj):
        df= dataframeObj.copy()
        ret = df[self.close].pct_change().fillna(0)
        ret_y = np.where(ret.shift(-1)>0, 1, 0)
        return ret_y

    def get_count_values(self, ret):

        return pd.Series(ret).value_counts()
    
    
    def get_dummies(self, dfObj):
        d_x = dfObj.copy()
        dummy_data = pd.get_dummies(d_x)
        return dummy_data

    """
    One-Hot Encoding
    Encoding refers to an approach that converts a categorical feature to a numerical vector. 
    One Hot Encoder is of the various techniques used to encode a categorical feature by assigning one binary column per category per categorical feature.

    pd.get_dummies() : The only advantage of pandas get_dummies() functions is its easy interpretability, and the fact that it returns a pandas dataframe with column names.

    """
    def get_hot_encoding_dummies(self, list_items):
        test_hours = pd.Series(list_items)

        return pd.get_dummies(test_hours)

    
    def get_encoded_data(self, df):

        # one hot encoding
        encoder = OneHotEncoder(sparse=False)

        # transform data - OHE takes 2D input
        onehot = encoder.fit_transform(df[[self.hours]])
        onehot


    

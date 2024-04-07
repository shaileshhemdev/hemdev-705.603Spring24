import pandas as pd

class Time_Series_Dataset:
    """
    A class used to represent the Time Series Dataset 

    ...

    Attributes
    ----------
    _df : df
        the dataframe representing the time series dataset
    class_label_col: str
        The name of the field that is working as class label 
    tot_train : df
        the dataframe comprising of training data for total transactions
    tot_test : df
        the dataframe comprising of testing data for total transactions
    fraud_train : df
        the dataframe comprising of training data for fraudulent transactions
    fraud_test : df
        the dataframe comprising of testing data for fraudulent transactions

    Methods
    -------
    get_training_dataset()
        Get Training data for the specified fold 
    get_testing_dataset()
        Get Testing data for the specified fold 
    """
    
    def __init__(self, transformed_df, class_label_col, train_size=209):
        """ Initializes the Fraud_Dataset Class

        Parameters
        ----------
        transformed_df : df
            The dataset (Pandas Dataframe) that represents the final cleansed and preprocessed data
        class_label_col : str
            The name of the column that holds the class label 

        """
        self._df = transformed_df
        self.class_label_col = class_label_col

        # Convert to dates
        self._df["ds"] = pd.to_datetime(self._df['ds'], format='%Y-%m-%d', errors='coerce')
        
        # Total Transactions 
        total_txn_group_df = self._df.groupby(pd.Grouper(key='ds',freq='W-Mon'))
        total_txn_agg_df = total_txn_group_df[['amt']].agg('sum')
        total_txn_agg_df['y'] = total_txn_group_df.size()
        total_txn_agg_df['ds'] = total_txn_agg_df.index

        # Fraudulent Transactions
        fraud_df = self._df[self._df[self.class_label_col] == 1] 
        fraud_txn_group_df = fraud_df.groupby(pd.Grouper(key='ds',freq='W-Mon'))
        fraud_txn_agg_df = fraud_txn_group_df[['amt']].agg('sum')
        fraud_txn_agg_df['y'] = fraud_txn_group_df.size()
        fraud_txn_agg_df['ds'] = fraud_txn_agg_df.index

        # Form the Training and Test sets with ~5 years for testing we will use the latest data for the best forecasting
        self.tot_train = total_txn_agg_df.iloc[:train_size]
        self.tot_test = total_txn_agg_df.iloc[train_size:]

        self.fraud_train = fraud_txn_agg_df.iloc[:train_size]
        self.fraud_test = fraud_txn_agg_df.iloc[train_size:]
    
    def get_training_dataset(self):
        """ Get the Training Dataset

        """

        # Return tuple of training data for Total and Fraudulent Transactions
        return (self.tot_train, self.fraud_train)

    def get_testing_dataset(self):
        """ Get the Testing Dataset

        """
        # Return tuple of training data for Total and Fraudulent Transactions
        return (self.tot_test, self.fraud_test)
    
       
    
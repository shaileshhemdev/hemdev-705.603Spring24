
from data_pipeline import ETL_Pipeline 
import pandas as pd
from prophet import Prophet

class Forecast_Model:
    """
    A class used to represent the Time Series Forecast Model 

    ...

    Attributes
    ----------


    Methods
    -------
    train()
        Train the data using training data
    test()
        Predict the test data

    """
    def __init__(self):
        """ Initializes the Time Series Model


        """
        self.model_total_transactions = Prophet()
        self.model_fraud_transactions = Prophet()

    def train(self, tot_train,fraud_train):
        """ Train the Model 

        Parameters
        ----------
        tot_train : df
            Dataframe holding the total transactions for training 
        fraud_train : df
            Dataframe holding the fraudulent transactions for training

        """
        self.model_total_transactions.fit(tot_train)
        self.model_fraud_transactions.fit(fraud_train)

    def test(self, tot_test, fraud_test):
        """ Test the Model 

        Parameters
        ----------
        tot_test : df
            Dataframe holding the fraudulent transactions for testing
        fraud_test : df
            Dataframe holding the fraudulent transactions for testing
        """
        # Find the predicted value
        future_total = self.model_total_transactions.make_future_dataframe(periods=len(tot_test),freq='W')
        future_fraud = self.model_fraud_transactions.make_future_dataframe(periods=len(fraud_test),freq='W')

        return (future_total, future_fraud)

    def predict(self, future_date_str):
        """ Predict whether the transaction is fraud depending on the transaction details

        Parameters
        ----------
        future_date : str
            Future Date provided
        """
        future_date_df = pd.DataFrame({'ds':[future_date_str]})
        future_date_df['ds'] = pd.to_datetime(future_date_df['ds'], format='%Y-%m-%d', errors='coerce')

        total_transactions = self.model_total_transactions.predict(future_date_df)
        fraud_transactions = self.model_fraud_transactions.predict(future_date_df)

        return (total_transactions['yhat'].values[0], fraud_transactions['yhat'].values[0])
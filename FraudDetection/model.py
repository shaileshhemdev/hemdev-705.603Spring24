from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from data_pipeline import ETL_Pipeline 
import pandas as pd
from metrics import Metrics

class Fraud_Detector_Model:
    """
    A class used to represent the Fraud Detection Model 

    ...

    Attributes
    ----------
    cls : RandomForestClassifier
        Classifier used for making prediction on whether a transaction is fraudulent or not

    Methods
    -------
    train()
        Train the data using training data
    test()
        Predict class for test data and return metrics 

    """
    def __init__(self, classifier_code='RF'):
        """ Initializes the Detection Model

        Parameters
        ----------
        classifier_type : str
            Enables using different classifiers. Currently following supported

            1. Random Forest (Default)
            2. Gradient Boosting
            3. ADA Boost

        """
        if (classifier_code == "RF"):
            self.cls = RandomForestClassifier(warm_start=True, max_depth=11, 
                                                   n_estimators=100, max_features=12, random_state=42)
        elif (classifier_code == "GB"):
            self.cls = GradientBoostingClassifier(n_estimators=100, max_features=12, learning_rate=0.1, 
                                                      max_depth=11, random_state=42)
        elif (classifier_code == "AB"):
            self.cls = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", learning_rate=0.1, random_state=42)
        else:
            self.cls = RandomForestClassifier(warm_start=True, max_depth=11, 
                                                   n_estimators=100, max_features=12, random_state=42)
        
    
    def train(self, X_train,y_train, addn_x=None, addn_y=None):
        """ Train the Model 

        Parameters
        ----------
        X_train : ndarray
            Data used for training the model specifically the features
        y_train : ndarray
            Data used for training the model specifically the class labels
        addn_x : ndarray
            Additional Optional Data used for training the model specifically the features (used 
            when training data is huge and needs some splitting)
        addn_y : ndarray
            Additional Optional Data used for training the model specifically the class labels 
            specifically the features (used when training data is huge and needs some splitting)
        """
        self.cls.fit(X_train, y_train)

        # If additional data is passed then use that for training
        if (addn_x is not None) & (addn_y is not None) :
            self.cls.n_estimators += 100
            self.cls.fit(addn_x, addn_y)

    def test(self, X_test, y_test):
        """ Test the Model 

        Parameters
        ----------
        X_test : ndarray
            Data used for testing the model specifically the features
        y_test : ndarray
            Data used for testing the model specifically the class labels
        """
        # Find the predicted value
        y_pred = self.cls.predict(X_test)

        # Initialize the metrics 
        metrics = Metrics()

        return metrics.run(y_test, y_pred)

    def predict(self, transaction_details):
        """ Predict whether the transaction is fraud depending on the transaction details

        Parameters
        ----------
        transaction_details : dictionary
            Dictionary of transaction attributes 
        """
        # Putting a dummy is_fraud to leverage the model
        transaction_details['is_fraud'] = 0

        # Create a dataframe from 
        input_df = pd.DataFrame.from_dict([transaction_details])

        # Run the data pipeline on the data
        dp = ETL_Pipeline('')
        transformed_df = dp.transform(input_df)

        # Extract the features in order to make the prediction
        X_predict = transformed_df.loc[:, transformed_df.columns != 'is_fraud'].values

        # Find the predicted value
        y_pred = self.cls.predict(X_predict)

        # Decide if its fraud
        is_fraud = False 
        if (y_pred == 1):
            is_fraud = True

        return is_fraud
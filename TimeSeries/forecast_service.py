from flask import Flask
from flask import request, jsonify
import sys
import os

from data_pipeline import ETL_Pipeline 
from dataset import Time_Series_Dataset
from model import Forecast_Model

app = Flask(__name__)

"""
A class used to provides an REST API to detect if a given transaction is fraudulent or not

As a part of service initialization we will read the training data, clean it, scale and encode it and then use it to train our model
...

Methods
-------

forecast()
    Will determine if the supplied transaction is fraudulent or not. To execute you use POST http://localhost:8786/detect-fraud

    Sample JSON as body below

    {
        "forecast_date": "2019-01-01"
    }

"""


@app.route('/fraud-forecast', methods=['POST'])
def forecast():
    """ Provides a forecast for the Total and Fraudulent Transactions
        
    Parameters 
    ----------
    Encoded into Json with following mandatory attributes

    forecast_date : str
        Date for which a forecast is needed


    Sample JSON Below

    {
        "forecast_date": "2019-01-01"
    }

    """
    # Obtain request payload as a dictionary
    forecast_details = request.json

    # Get the forecasted date
    forecast_date = forecast_details["forecast_date"]

    # Pass dictionary data to model's prediction 
    total_transactions, fraudulent_transactions = model.predict(forecast_date)

    # Return the result as Json
    return jsonify({"total_transactions":total_transactions, "fraudulent_transactions":fraudulent_transactions})

if __name__ == "__main__":
    """ Initializes the Forecast Service Class

    It does the following as a part of initialization

    1. Initialize the ETL_Pipeline and use it to process the training data to get the features we need
    2. Initialize Time_Series_Dataset to get the training data split
    3. Initialize the Forecast_Model to train the classifier on the training data  
    """
    flaskPort = 8786

    # Get command line arguments
    if (len(sys.argv)>1):
        data_folder                 = sys.argv[1]
        fraud_training_data_file    = sys.argv[2]
    else: 
        data_folder = os.environ['data-folder']
        fraud_training_data_file = os.environ['training-data-file']

    # Process the Data needed to train the model
    print(f'Start an ETL_Pipeline to load training data with shared folder = {data_folder} and training data file = {fraud_training_data_file}')
    dp = ETL_Pipeline(data_folder)
    df = dp.process(fraud_training_data_file)
    print('Successfully processed and created feature data and initialized metrics')

    # Create a fraud dataset with single fold
    fd = Time_Series_Dataset(df,'is_fraud')
    print('Successfully created Fraud Dataset')

    # Obtain the training data
    tot_train, fraud_train = fd.get_training_dataset()
    tot_test, fraud_test = fd.get_testing_dataset()
    print('Successfully created training and testing data')

    # Train the Model
    model = Forecast_Model()
    print('Successfully created Fraud Data Model')

    model.train(tot_train, fraud_train)
    print('Successfully trained Forecasting Model')

    # Now that all the setup has been done start the service
    print('Starting Server...')
    app.run(host = '0.0.0.0', port = flaskPort)


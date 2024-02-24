from flask import Flask
from flask import request, jsonify
import sys

from data_pipeline import ETL_Pipeline 
from dataset import Fraud_Dataset
from model import Fraud_Detector_Model
from metrics import Metrics

app = Flask(__name__)

"""
A class used to provides an REST API to detect if a given transaction is fraudulent or not

As a part of service initialization we will read the training data, clean it, scale and encode it and then use it to train our model
...

Methods
-------
getStats()
    Will run the testing data through the model to return the metrics. To execute you use GET http://localhost:8786/stats
detect_fraud()
    Will determine if the supplied transaction is fraudulent or not. To execute you use POST http://localhost:8786/detect-fraud

    Sample JSON as body below

    {
        "trans_date_trans_time": "2019-01-01 00:00:18",
        "cc_num": "2703186189652095",
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "misc_net",
        "amt": 4.97,
        "first": "John",
        "last": "Doe",
        "sex": "F",
        "street": "57636 Russet Ln",
        "city": "South Lyon",
        "state": "MI",
        "zip": "48122",
        "lat": 36.079, 
        "long": -81.178,
        "city_pop": 2309,
        "job": "Psychologist, counselling",  
        "dob": "1988-03-09",      
        "trans_num": "0b242abb623afc578575680df30655b9",
        "unix_time" : 1325376018,
        "merch_lat": 36.011,
        "merch_long": -82.048
    }

"""

@app.route('/stats', methods=['GET'])
def getStats():
    """ Provides statistics for the Fraud model used to make the prediction
        
    Returns
    ----------
    Json with following attributes

    Accuracy : float
        Accuracy of the test results. Not the best metric for this dataset
    Balanced Accuracy : float
        Balanced Accuracy of the test results. Takes into account the imbalance in the dataset
    Specificity : float
        Calculated as True negatives / (True negatives + False Positives) - a good measure of how many times we got valid transactions right
    Sensitivity : float
        Calculated as True positives / (True positives + False negatives) - a good measure of how many times we got fraudulent transactions right
    Precision : float
        A good measure of how many times we got fraudulent transactions right
    Recall : float
        A good measure of how many times we got valid transactions right
    F1 Score : float
        Harmonic Mean of Precision and Recall. A good metric for this model
    ROC AUC Score: float
        A good metric for this model as it measures optimality with precision and recall
    Average Precision Score : float
        A good alternatives for ROC AUC if imbalance is high

    Sample JSON Below

    {
        "Accuracy": 0.9975869064572656,
        "Average Precision Score": 0.5495803576746265,
        "Balanced Accuracy": 0.8121015145716799,
        "F1 Score": 0.7295171245310419,
        "Precision": 0.8766724840023269,
        "ROC AUC Score": 0.8121015145716798,
        "Recall": 0.6246632124352332,
        "Sensitivity": 0.6246632124352332,
        "Specificity": 0.9995398167081265
    }

    """
    # Obtain the metrics 
    acc, acc_bal, specificity, sensitivity, prec, recall, f1, roc_auc, avg_prec = model.test(X_test, y_test)

    # Create a dictionary for statistics
    statistics = {}
    statistics['Accuracy'] = acc
    statistics['Balanced Accuracy'] = acc_bal
    statistics['Specificity'] = specificity
    statistics['Sensitivity'] = sensitivity
    statistics['Precision'] = prec
    statistics['Recall'] = recall
    statistics['F1 Score'] = f1
    statistics['ROC AUC Score'] = roc_auc
    statistics['Average Precision Score'] = avg_prec

    # Return the statistics encoded as Json
    return jsonify(statistics)

@app.route('/detect-fraud', methods=['POST'])
def detect_fraud():
    """ Detects if a given transaction is fraudulent or not
        
    Parameters 
    ----------
    Encoded into Json with following mandatory attributes

    trans_date_trans_time : str
        Transaction date and time
    cc_num : str
        Unique customer number/ID
    merchant : str
        Merchant/vendor name for transaction
    category : str
        Category of purchase (e.g., entertainment, gas_transport, food_dining, etc.)
    amt : float
        Total amount of transaction
    first : str
        Customer first name
    last: float
        Customer last name
    sex : str
        Customer's sex
    street : str
        Street address of customer
    city : str
        City address of customer
    state : str
        State of customer residency
    zip : str
        Zip code of customer
    lat : float
        latitude coordinate of customer address
    long : float
        longitude coordinate of customer address
    city_pop : int
        Population of city
    job : str
        Customer's employment title
    dob : str
        Customer's date of birth
    trans_num : str
        Unique transaction number
    unix_time : int
        Timestamp of transaction
    merch_lat : float
        Latitude of merchant/vendor
    merch_long : str
        Longitude of merchant/vendor

    Sample JSON Below

    {
        "trans_date_trans_time": "2019-01-01 00:00:18",
        "cc_num": "2703186189652095",
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "misc_net",
        "amt": 4.97,
        "first": "John",
        "last": "Doe",
        "sex": "F",
        "street": "57636 Russet Ln",
        "city": "South Lyon",
        "state": "MI",
        "zip": "48122",
        "lat": 36.079, 
        "long": -81.178,
        "city_pop": 2309,
        "job": "Psychologist, counselling",  
        "dob": "1988-03-09",      
        "trans_num": "0b242abb623afc578575680df30655b9",
        "unix_time" : 1325376018,
        "merch_lat": 36.011,
        "merch_long": -82.048
    }

    """
    # Obtain request payload as a dictionary
    transaction_details = request.json

    # Pass dictionary data to model's prediction to detect if fraud or not
    is_fraud = model.predict(transaction_details)

    # Return the result as Json
    return jsonify({"is_fraud":is_fraud})

if __name__ == "__main__":
    """ Initializes the Fraud Service Class

    It does the following as a part of initialization

    1. Initialize the ETL_Pipeline and use it to process the training data to get the features we need
    2. Initialize Fraud_Dataset to get the training data split
    3. Initialize the Metrics class used to generate and provide latest statistics
    4. Initialize the Fraud_Detector_Model to train the classifier on the training data  
    """
    flaskPort = 8786

    # Get command line arguments
    data_folder                 = sys.argv[1]
    fraud_training_data_file    = sys.argv[2]

    # Process the Data needed to train the model
    print(f'Start an ETL_Pipeline to load training data with shared folder = {data_folder} and training data file = {fraud_training_data_file}')
    dp = ETL_Pipeline(data_folder)
    df = dp.process(fraud_training_data_file)

    # Initialize the metrics
    print('Successfully processed and created feature data and initialized metrics')
    metrics = Metrics()

    # Create a fraud dataset with single fold
    fd = Fraud_Dataset(df,'is_fraud',2)
    print('Successfully created Fraud Dataset')

    # Obtain the training data
    X_train, y_train = fd.get_training_dataset(0)
    X_test, y_test = fd.get_testing_dataset(0)
    X_val, y_val = fd.get_validation_dataset(0)
    print('Successfully created training and testing data using K-Fold')

    # Train the Model
    model = Fraud_Detector_Model()
    print('Successfully created Fraud Data Model')

    model.train(X_train, y_train, X_val, y_val)
    print('Successfully trained Fraud Model')

    # Now that all the setup has been done start the service
    print('Starting Server...')
    app.run(host = '0.0.0.0', port = flaskPort)


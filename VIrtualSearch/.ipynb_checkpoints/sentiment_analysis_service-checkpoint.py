from flask import Flask
from flask import request, jsonify
import sys
import os

from data_pipeline import Text_Pipeline 
from dataset import Sentiment_Analysis_Dataset
from model import Sentiment_Analysis_Model
from metrics import Metrics

app = Flask(__name__)

"""
A class used to provides an REST API to get the sentiment for a text

As a part of service initialization we will read the training data, clean it, scale and encode it and then use it to train our model
...

Methods
-------
getStats()
    Will run the testing data through the model to return the metrics. To execute you use GET http://localhost:8786/stats
get_sentiment()
    Will determine sentiment associated with the review. To execute you use POST http://localhost:8786/get-sentiment

    Sample JSON as body below

    {
        "reviews": ["I loved it", "I did not like it", "It was OK"]
    }

"""

@app.route('/stats', methods=['GET'])
def getStats():
    """ Provides statistics for the Sentiment Analysis model used to make the prediction
        
    Returns
    ----------
    Json with following attributes

    Accuracy : float
        Accuracy of the test results. Not the best metric for this dataset
    Balanced Accuracy : float
        Balanced Accuracy of the test results. Takes into account the imbalance in the dataset
    Specificity : float
        Calculated as True negatives / (True negatives + False Positives) - a good measure of how many times we got sentiment right
    Sensitivity : float
        Calculated as True positives / (True positives + False negatives) - a good measure of how many times we got sentiment right
    Precision : float
        A good measure of how many times we got sentiment right
    Recall : float
        A good measure of how many times we got sentiment right
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

@app.route('/get-sentiment', methods=['POST'])
def detect_fraud():
    """ Detects sentiment associated
        
    Parameters 
    ----------
    Encoded into Json with following mandatory attributes

    reviews : list
        Array of text for which sentiment needs to be obtained


    Sample JSON Below

    {
        "reviews": ["I loved it", "I did not like it", "It was OK"]
    }

    """
    # Obtain request payload as a dictionary
    review_details = request.json

    # Pass dictionary data to model's prediction to detect if fraud or not
    text_reviews = model.predict(review_details["reviews"])

    # Return the result as Json
    return jsonify(text_reviews)

if __name__ == "__main__":
    """ Initializes the Sentiment Analysis Service Class

    It does the following as a part of initialization

    1. Initialize the Text_Pipeline and use it to process the training data to get the features we need
    2. Initialize Sentiment_Analysis_Dataset to get the training data split
    3. Initialize the Metrics class used to generate and provide latest statistics
    4. Initialize the Sentiment_Analysis_Model to train the classifier on the training data  
    """
    flaskPort = 8786

    # Get command line arguments
    if (len(sys.argv)>1):
        data_folder                 = sys.argv[1]
        training_data_file    = sys.argv[2]
    else: 
        data_folder = os.environ['data-folder']
        training_data_file = os.environ['training-data-file']

    # Process the Data needed to train the model
    print(f'Start an ETL_Pipeline to load training data with shared folder = {data_folder} and training data file = {training_data_file}')
    #dp = Text_Pipeline(data_folder)
    #df = dp.process(fraud_training_data_file)

    # Initialize the metrics
    print('Successfully processed and created feature data and initialized metrics')
    metrics = Metrics()

    # Create a fraud dataset with single fold
    #fd = Fraud_Dataset(df,'is_fraud',2)
    print('Successfully created Fraud Dataset')

    # Obtain the training data
    #X_train, y_train = fd.get_training_dataset(0)
    #X_test, y_test = fd.get_testing_dataset(0)
    #X_val, y_val = fd.get_validation_dataset(0)
    print('Successfully created training and testing data using K-Fold')

    # Train the Model
    #model = Fraud_Detector_Model()
    print('Successfully created Fraud Data Model')

    #model.train(X_train, y_train, X_val, y_val)
    print('Successfully trained Fraud Model')

    # Now that all the setup has been done start the service
    print('Starting Server...')
    app.run(host = '0.0.0.0', port = flaskPort)


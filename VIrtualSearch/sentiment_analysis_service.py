from flask import Flask
from flask import request, jsonify
import sys
import os

from transformers import pipeline
from data_pipeline import Text_Pipeline 
from etl_pipeline import ETL_Pipeline
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
    Precision : float
        A good measure of how many times we got sentiment right
    Recall : float
        A good measure of how many times we got sentiment right
    F1 Score : float
        Harmonic Mean of Precision and Recall. A good metric for this model

    Sample JSON Below

    {
        "Accuracy": 0.9975869064572656,
        "Balanced Accuracy": 0.8121015145716799,
        "F1 Score": 0.7295171245310419,
        "Precision": 0.8766724840023269,
        "Recall": 0.6246632124352332
    }

    """
    # Obtain the metrics 
    acc, acc_bal, prec, recall, f1,  = sentiment_model.test(X_test, y_test)

    # Create a dictionary for statistics
    statistics = {}
    statistics['Accuracy'] = acc
    statistics['Balanced Accuracy'] = acc_bal
    statistics['Precision'] = prec
    statistics['Recall'] = recall
    statistics['F1 Score'] = f1

    # Return the statistics encoded as Json
    return jsonify(statistics)

@app.route('/get-sentiment', methods=['POST'])
def get_sentiment():
    """ Get sentiment associated
        
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
    text_reviews = sentiment_model.predict(review_details["reviews"])

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
        data_folder           = sys.argv[1]
        training_data_file    = sys.argv[2]
    else: 
        data_folder = os.environ['data-folder']
        training_data_file = os.environ['training-data-file']

    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_pipe = pipeline("sentiment-analysis", model=model_id)

    # Process the Data needed to train the model
    print(f'Start an ETL_Pipeline to load training data with shared folder = {data_folder} and training data file = {training_data_file}')
    dp = ETL_Pipeline(data_folder)
    df = dp.process(training_data_file)

    # Initialize the metrics
    print('Successfully processed and created feature data and initialized metrics')
    metrics = Metrics()

    # Create a reviews dataset with single fold
    td = Sentiment_Analysis_Dataset(df,'class',5)
    print('Successfully created Reviews Dataset')

    # Obtain the training data
    X_train, y_train = td.get_training_dataset(0)
    X_test, y_test = td.get_testing_dataset(0)
    X_val, y_val = td.get_validation_dataset(0)
    print('Successfully created training and testing data using K-Fold')

    # Initialize the Model
    sentiment_model = Sentiment_Analysis_Model(classifier_code='TWITTER-RB', model_param=sentiment_pipe)
    print('Successfully created Sentiment Analysis Model')

    #model.train(X_train, y_train, X_val, y_val)
    #print('Successfully trained Sentiment Analysis Model')

    # Now that all the setup has been done start the service
    print('Starting Server...')
    app.run(host = '0.0.0.0', port = flaskPort)



from transformers import pipeline
from data_pipeline import Text_Pipeline 
import pandas as pd
from metrics import Metrics

class Sentiment_Analysis_Model:
    """
    A class used to represent the Sentiment Analysis Model 

    ...

    Attributes
    ----------
    cls : 
        Classifier used for making prediction on the sentiment 

    Methods
    -------
    train()
        Train the data using training data
    test()
        Predict class for test data and return metrics 

    """
    def __init__(self, classifier_code='TWITTER-RB', model_param=None):
        """ Initializes the Sentiment Analysis Model

        Parameters
        ----------
        classifier_type : str
            Enables using different classifiers. Currently following supported

            1. TWITTER-RB = twitter-roberta-base-sentiment-latest
            2. 

        """
        if (model_param is not None):
            self.cls = model_param
            self.model_type = 'TRANSFORMER_PRETRAINED_LLM'
        elif (classifier_code == "TWITTER-RB"):
            model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            sentiment_pipe = pipeline("sentiment-analysis", model=model_id)
            self.cls = sentiment_pipe
            self.model_type = 'TRANSFORMER_PRETRAINED_LLM'
        else:
            model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            sentiment_pipe = pipeline("sentiment-analysis", model=model_id)
            self.cls = sentiment_pipe
            self.model_type = 'TRANSFORMER_PRETRAINED_LLM'
        
        # Setup the text pipeline
        self.text_pipeline = Text_Pipeline('CONVERT')
        
        # Create mapped sentiments
        self.mapped_sentiments = {'positive':2, 'neutral':1, 'negative':0}
    
    def train(self, X_train,y_train):
        """ Train the Model 

        Parameters
        ----------
        X_train : ndarray
            Data used for training the model specifically the features
        y_train : ndarray
            Data used for training the model specifically the class labels
        """
        self.cls.fit(X_train, y_train)

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
        if (self.model_type == 'TRANSFORMER_PRETRAINED_LLM'):
            y_pred = self.obtain_sentiment(X_test)

        # Initialize the metrics 
        metrics = Metrics()

        return metrics.run(y_test, y_pred)

    def obtain_sentiment(self, X_test):
        """ Obtain the sentiment encoded as class label values

        Parameters
        ----------
        X_test : ndarray
            Data used for testing the model specifically the text
        """
        # Preprocess the data
        total_size = len(X_test)

        # Obtain sentiment scores for all the reviews
        sentiment_scores = []
        end = 100
        for i in range(0,total_size,100):
            # Create a Pandas series 
            s = pd.Series(X_test[i:end]) 

            # Obtain pre processed series
            preprocessed_series = self.text_pipeline.preprocess(s)

            # Get reviews
            preprocessed_text = preprocessed_series.values.tolist()
            
            # Analyze sentiment for each text
            sentiments = [self.analyze_sentiment(text[:2000]) for text in preprocessed_text]

            # Append to overall scores
            sentiment_scores += sentiments
            end += 100

        return [self.mapped_sentiments[s] for s in sentiment_scores]

    def analyze_sentiment(self, text):
        """ Function that analyzes the sentiment

        Parameters
        ----------
        text : str
            The text for which sentiment needs to be derived
        """
        result = self.cls(text)
        return result[0]['label']

    def predict(self, text_array):
        """ Predict whether the transaction is fraud depending on the transaction details

        Parameters
        ----------
        text_array : list
            List of text input
        """
        # Create a series
        s = pd.Series(text_array) 

        # Obtain pre processed series
        preprocessed_series = self.text_pipeline.preprocess(s)

        # Get reviews
        preprocessed_text = preprocessed_series.values.tolist()

        print(preprocessed_text)
            
        # Analyze sentiment for each text
        sentiments = [self.analyze_sentiment(text[:2000]) for text in text_array]
        
        # Return mapped class values
        return [self.mapped_sentiments[s] for s in sentiments]
    
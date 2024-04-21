import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

class Metrics:
    """
    A class used to provides Metrics Services

    ...

    Methods
    -------
    generate_report()
        Will generate a report for all the folds for which the model was run
    run()
        Obtain the following metrics
        1. Accuracy
        2. Balanced Accuracy
        3. Precision
        4. Recall
        5. F1 Score
        6. ROC AUC Score
        7. Average Precision Score

    """
    def __init__(self):
        """ Initializes the Metrics Class
        """

    def generate_report(self, accs, acc_bals, precs, recalls, f1s, classifiers, report_file):
        """ Generates a Report and Stores it in the results directory

        Parameters
        ----------
        acc : float
            Accuracy of the test results. Not the best metric for this dataset
        acc_bal : float
            Balanced Accuracy of the test results. Takes into account the imbalance in the dataset
        Precision : float
            A good measure of how many times we got fraudulent transactions right
        Recall : float
            A good measure of how many times we got valid transactions right
        F1 Score : float
            Harmonic Mean of Precision and Recall. A good metric for this model
        classifiers : ndarray
            List of classifiers for which results are being reported
        report_file : str
            The full path to the file where the results need to be stored

        """

        # Open a file
        f = open(report_file, "w")

        i = 0
        for cls in classifiers:
            acc         = accs[i]
            acc_bal     = acc_bals[i]
            prec        = precs[i]
            recall      = recalls[i]
            f1          = f1s[i]

            i = i + 1

            f.write(f"Model Results for {cls}:\n")
            f.write(f"\t\tAccuracy = {acc:.2%}\n")
            f.write(f"\t\tBalanced Accuracy = {acc_bal:.2%}\n")
            f.write(f"\t\tPrecision = {prec:.2%}\n")
            f.write(f"\t\tRecall = {recall:.2%}\n")
            f.write(f"\t\tF1 Score = {f1:.2%}\n")
            f.write("\n")

        f.close() 

    def run(self, y_val, y_pred):
        """ Calculates the various metrics

        Parameters
        ----------
        y_val : ndarray
            The ground truth for the data
        y_pred : ndarray
            The predicted values for the observations
        dataset : str
            The type of dataset. Defaults to 'Testing'

        Returns
        ----------
        acc : float
            Accuracy of the test results. Not the best metric for this dataset
        acc_bal : float
            Balanced Accuracy of the test results. Takes into account the imbalance in the dataset
        Precision : float
            A good measure of how many times we got fraudulent transactions right
        Recall : float
            A good measure of how many times we got valid transactions right
        F1 Score : float
            Harmonic Mean of Precision and Recall. A good metric for this model

        """

        # Initialize Metric Arrays 
        acc             = accuracy_score(y_val, y_pred) 
        balanced_acc    = balanced_accuracy_score(y_val, y_pred) 
        precision       = precision_score(y_val, y_pred, average='weighted')
        recall          = recall_score(y_val, y_pred, average='weighted')
        f1              = f1_score(y_val, y_pred, average='weighted')

        return (acc, balanced_acc, precision, recall, f1)
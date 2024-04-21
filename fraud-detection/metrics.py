import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
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
        3. Specificity
        4. Sensitivity
        5. Precision
        6. Recall
        7. F1 Score
        8. ROC AUC Score
        9. Average Precision Score

    """
    def __init__(self):
        """ Initializes the Metrics Class
        """

    def generate_report(self, accs, acc_bals, specificitys, sensitivitys, precs, recalls, f1s, roc_aucs, avg_precs, classifiers, report_file):
        """ Generates a Report and Stores it in the results directory

        Parameters
        ----------
        acc : float
            Accuracy of the test results. Not the best metric for this dataset
        acc_bal : float
            Balanced Accuracy of the test results. Takes into account the imbalance in the dataset
        specificity : float
            Calculated as True negatives / (True negatives + False Positives) - a good measure of how many times we got valid transactions right
        sensitivity : float
            Calculated as True positives / (True positives + False negatives) - a good measure of how many times we got fraudulent transactions right
        Precision : float
            A good measure of how many times we got fraudulent transactions right
        Recall : float
            A good measure of how many times we got valid transactions right
        F1 Score : float
            Harmonic Mean of Precision and Recall. A good metric for this model
        ROC AUC : float
            A good metric for this model as it measures optimality with precision and recall
        Average Precision Score : float
            A good alternatives for ROC AUC if imbalance is high
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
            specificity = specificitys[i]
            sensitivity = sensitivitys[i]
            prec        = precs[i]
            recall      = recalls[i]
            f1          = f1s[i]
            roc_auc     = roc_aucs[i]
            avg_prec    = avg_precs[i]

            i = i + 1

            f.write(f"Model Results for {cls}:\n")
            f.write(f"\t\tAccuracy = {acc:.2%}\n")
            f.write(f"\t\tBalanced Accuracy = {acc_bal:.2%}\n")
            f.write(f"\t\tSpecificity = {specificity:.2%}\n")
            f.write(f"\t\tSensitivity = {sensitivity:.2%}\n")
            f.write(f"\t\tPrecision = {prec:.2%}\n")
            f.write(f"\t\tRecall = {recall:.2%}\n")
            f.write(f"\t\tF1 Score = {f1:.2%}\n")
            f.write(f"\t\tROC AUC Score = {roc_auc:.2%}\n")
            f.write(f"\t\tAverage Precision Score = {avg_prec:.2%}\n")
            f.write("\n")

        f.close() 

    def run(self, y_val, y_pred, dataset='Testing'):
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
        specificity : float
            Calculated as True negatives / (True negatives + False Positives) - a good measure of how many times we got valid transactions right
        sensitivity : float
            Calculated as True positives / (True positives + False negatives) - a good measure of how many times we got fraudulent transactions right
        Precision : float
            A good measure of how many times we got fraudulent transactions right
        Recall : float
            A good measure of how many times we got valid transactions right
        F1 Score : float
            Harmonic Mean of Precision and Recall. A good metric for this model
        ROC AUC : float
            A good metric for this model as it measures optimality with precision and recall
        Average Precision Score : float
            A good alternatives for ROC AUC if imbalance is high
        """
        # Get the core metrics from confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

        # Initialize Metric Arrays 
        acc             = accuracy_score(y_val, y_pred) 
        balanced_acc    = balanced_accuracy_score(y_val, y_pred) 
        specificity     = tn / (tn + fp)
        sensitivity     = tp / (tp + fn)
        precision       = precision_score(y_val, y_pred)
        recall          = recall_score(y_val, y_pred)
        f1              = f1_score(y_val, y_pred)
        roc_auc         = roc_auc_score(y_val, y_pred)
        avg_prec_score  = average_precision_score(y_val, y_pred)

        return (acc, balanced_acc, specificity, sensitivity, precision, recall, f1, roc_auc, avg_prec_score)
import numpy as np
from sklearn.metrics import accuracy_score
from Levenshtein import distance as lev

class Metrics:
    """
    A class used to provides Metrics Services

    ...

    Methods
    -------
    generate_report()
        Will generate a report for all the license plates for which the model was run
    run()
        Obtain the following metrics
        1. Accuracy
        2. Precision
        3. Average Levenshtein
        4. Std Deviation Levenshtein

    """
    def __init__(self):
        """ Initializes the Metrics Class
        """

    def generate_report(self, license_plate_numbers, report_file, model_results):
        """ Generates a Report and Stores it in the results directory

        Parameters
        ----------
        license_plate_numbers : ndarray
            List of actual license plate numbers
        report_file : str
            The full path to the file where the results need to be stored
        metrics : list
            List of metrics for each license plate number
        """

        # Open a file
        f = open(report_file, "w")

        i = 0
        for lp in license_plate_numbers:
            lp_metric   = self.run(lp,model_results[i][2],model_results[i][0])
            acc         = lp_metric[0]
            prec        = lp_metric[1]
            lev_mean    = lp_metric[2]
            lev_std    = lp_metric[3]

            i = i + 1

            f.write(f"Model Results for {lp}:\n")
            f.write(f"\t\tAccuracy = {acc:.2%}\n")
            f.write(f"\t\tPrecision = {prec:.2%}\n")
            f.write(f"\t\tMean Levenshtein = {lev_mean:.2f}\n")
            f.write(f"\t\tStd Levenshtein = {lev_std:.2f}\n")
            f.write("\n")

        f.close() 

    def run(self, y_val, y_pred, y_pred_best):
        """ Calculates the various metrics

        Parameters
        ----------
        y_val : ndarray
            The ground truth for the data
        y_pred : ndarray
            The predicted set of values from all the images

        Returns
        ----------
        acc : float
            Final accuracy of the best prediction
        Precision : float
            A good measure of how many times we got the correct license plate number
        lev_mean : float
            Mean for the Levenshtein distance
        lev_std : float
            Standard deviation for the Levenshtein distance
        """
        # Get a list of repeat values for 
        y_actuals = np.repeat(y_val, len(y_pred))

        # Compute the precision & average levinstein distance
        lev_list = list()
        tp, fp = 0, 0
        i = 0
        for y in y_pred:
            lev_list.append(self.get_distance(y_actuals[i],y))

            # Calculate the true positives and false positives
            if (y == y_val):
                tp += 1
            else:
                fp += 1

            i += 1
        if (len(lev_list)==0):
            lev_list.append(self.get_distance(y_val,y_pred_best))

        # Get Average Lev for all predictions
        lev_mean = np.mean(np.array(lev_list))
        lev_std = np.mean(np.array(lev_list))

        # Compute additional metrics
        acc             = accuracy_score(np.array([y_val]), np.array([y_pred_best])) 
        if (len(y_pred)==0):
            precision = 0
        else:
            precision       = tp / (tp + fp)

        return (acc, precision, lev_mean, lev_std)
    
    def get_distance(self, actual, predicted):
       """ Calculates the distance between the actual and prediced license plate

        Parameters
        ----------
        actual : str
            The actual license plate number
        predicted : str
            The predictedlicense plate number

        Returns
        ----------
        dist : int
            Levenshtein Distance
        """
       return lev(predicted, actual)

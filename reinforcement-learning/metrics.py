import numpy as np


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
        1. Average Conversion Rate
        2. Median Conversion Rate
        3. Minimum Conversion Rate
        4. Maximum Conversion Rate

    """
    def __init__(self):
        """ Initializes the Metrics Class
        """

    def generate_report(self, folds, average_conversions, median_conversions, min_conversions, max_conversions, report_file):
        """ Generates a Report and Stores it in the results directory

        Parameters
        ----------
        folds : int
            No of folds given a K Fold Strategy
        average_conversions : ndarray
            Array of by fold average conversion rate achieved
        median_conversions : ndarray
            Array of by fold median conversion rate achieved
        min_conversions : ndarray
            Array of by fold minimum conversion rate achieved
        max_conversions : ndarray
            Array of by fold maximum conversion rate achieved
        """

        # Open a file
        f = open(report_file, "w")

        i = 0
        for fold in folds:
            avg         = average_conversions[i]
            median      = median_conversions[i]
            min         = min_conversions[i]
            max         = max_conversions[i]

            i = i + 1

            f.write(f"Model Results for {fold}:\n")
            f.write(f"\t\tAverage Conversion Rate = {avg:.2%}\n")
            f.write(f"\t\tMedian Conversion Rate  = {median:.2%}\n")
            f.write(f"\t\tMinimum Conversion Rate  = {min:.2%}\n")
            f.write(f"\t\tMaximum Conversion Rate  = {max:.2%}\n")
            f.write("\n")

        f.close() 

    def run(self, conversion_rates):
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
        average_conversion_rate : float
            Calculates the Average Conversion Rate achieved for a particular fold
        median_conversion_rate : float
            Calculates the Median Conversion Rate achieved for a particular fold
        min_conversion_rate : float
            Calculates the Minimum Conversion Rate achieved for a particular fold
        max_conversion_rate : float
            Calculates the Maximum Conversion Rate achieved for a particular fold
        """
        # Initialize Metrics
        average_conversion_rate     = np.mean(conversion_rates) 
        median_conversion_rate      = np.median(conversion_rates) 
        min_conversion_rate         = np.min(conversion_rates) 
        max_conversion_rate         = np.max(conversion_rates) 

        return (average_conversion_rate, median_conversion_rate, min_conversion_rate, max_conversion_rate)
    
    def state_decoded(self, state):
        """ Calculates the decoded state

        Parameters
        ----------
        state : int
            The state value encoded 

        Returns
        ----------
        state : tuple
            Returns the decoded state
        """
        subject = (state // 5880) + 1
        rem = state % 5880
        
        dow = rem // 840
        rem = rem % 840
        
        tg = rem // 120
        rem = rem % 120
        
        ed = rem // 20
        rem = rem % 20
        
        ag = rem // 4
        rem = rem % 4
        
        gender = rem // 2
        type_val = rem % 2
        
        return (subject,dow,tg,ed,ag,gender,type_val)
    
    def conversion_rates(self, target_df, candidate_states):
        conversions = []
        for i in range(len(candidate_states)):
            state = self.state_decoded(candidate_states[i])
            
            filtered_df = target_df[(target_df['SubjectLine_ID'] == state[0]) & (target_df['Sent_Day'] == state[1]) & (target_df['Tenure_Group'] == state[2])
                            & (target_df['Email_Domain'] == state[3]) & (target_df['Age_Group'] == state[4]) & (target_df['Gender'] == state[5]) 
                            & (target_df['Type'] == state[6])]
            responses_received = np.sum(filtered_df['Response_Received'].values)
            emails_sent = np.sum(filtered_df['Sent_Emails'].values)

            if (emails_sent > 0):
                conversions += [responses_received/emails_sent]
            else:
                conversions += [0.0]
        
        return conversions
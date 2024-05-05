from data_pipeline import ETL_Pipeline 
from email_campaign_field import EmailCampaignField
import pandas as pd
from metrics import Metrics
import random
import numpy as np

class Email_Campaign_Model:
    """
    A class used to represent the Email Campaign Model 

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
    def __init__(self, email_campaign_df, starting_state = (1,0,0,0,0,0,0), threshold_sent_emails = 10, threshold_conversion_rate = 0.3, 
                 qtable_file=None, states=None):
        """ Initializes the Email Campaign Reinforcement Model

        Parameters
        ----------
        email_campaign_df : df
            Dataframe representing the Email Campaign Data
        threshold_sent_emails : int
            Threshold of sent emails to consider before stopping next action
        threshold_conversion_rate : float
            Threshold of Conversion Rate to consider before stopping next action
        """
        self.email_campaign_df = email_campaign_df
        self.states = states
        self.threshold_sent_emails = threshold_sent_emails
        self.threshold_conversion_rate = threshold_conversion_rate
        email_campaign_field = EmailCampaignField(self.email_campaign_df,starting_state, self.states,
                                                  self.threshold_conversion_rate, self.threshold_sent_emails)
        self.states = email_campaign_field.get_states()
        self.number_of_states =  email_campaign_field.get_number_of_states()

        if (qtable_file is not None):
            self.q_table_df = pd.read_csv(qtable_file)
    
    def get_states(self):
        return self.states

    def train(self, iterations=20000, starting_state = (1,0,0,0,0,0,0), 
              epsilon = 0.1, alpha = 0.1,gamma = 0.6):
        """ Train the Model 

        Parameters
        ----------
        epsilon : float
            Controls whether you explore or exploit
        alpha : float
            Modulates the reward
        gamma : float

        """

        # Use states from the last attempt to get number of states and actions needed for Q Table Sizing
        number_of_actions = 7

        # Initialize the Q Table
        q_table = np.zeros((self.number_of_states, number_of_actions))

        # Run several iterations to cover the state space to generate a rich Q Table
        for _ in range(iterations):
            email_campaign_field = EmailCampaignField(self.email_campaign_df,starting_state, self.states,
                                                      self.threshold_conversion_rate, self.threshold_sent_emails)
            done = False
            #print(email_campaign_field.get_state())
            # Keep trying different actions till we reach conversion rate for at least 1 email subject
            while not done:
                state = email_campaign_field.get_state()
                if random.uniform(0,1) < epsilon:
                    action = random.randint(0,6) # Explore
                else:
                    action = np.argmax(q_table[state]) # Exploit

                reward, done = email_campaign_field.make_action(action)
                new_state = email_campaign_field.get_state()
                new_state_max = np.max(q_table[new_state])

                q_table[state, action] = (1-alpha)*q_table[state, action] + alpha*(reward + gamma*new_state_max - q_table[state, action])
            
            # Pick a random starting state so that we can cover the entire state space
            starting_state = (random.randint(1,3),random.randint(0,6),random.randint(0,6),random.randint(0,5),
                                random.randint(0,4),random.randint(0,1),random.randint(0,1))
        
        q_table_df = pd.DataFrame(data=q_table, columns=['Day of Week','Tenure Group','Email Domain','Age Group',
                                                'Gender','Type','Subject Id']) 
        self.q_table_df = q_table_df
        return q_table_df 

    def get_candidate_states(self, starting_state):
        q_table = self.q_table_df[["Day of Week","Tenure Group","Email Domain","Age Group","Gender","Type","Subject Id"]].values
        epsilon = 0 # Assumes learning is over
        alpha = 0.1
        gamma = 0.6
        
        email_campaign_field = EmailCampaignField(self.email_campaign_df,starting_state, self.states,self.threshold_conversion_rate,self.threshold_sent_emails)
        done = False
        steps = 0
        candidate_states = []
        while not done:
            state = email_campaign_field.get_state()
            if random.uniform(0,1) < epsilon:
                action = random.randint(0,5) #Explore
            else:
                action = np.argmax(q_table[state]) #Exploit
            
            reward, done = email_campaign_field.make_action(action)

            new_state = email_campaign_field.get_state()
            new_state_max = np.max(q_table[new_state])
            
            candidate_states += [new_state]
            q_table[state, action] = (1-alpha)*q_table[state, action]+alpha*(reward+gamma*new_state_max - q_table[state, action])

            steps = steps + 1
            #print(steps)
        #print(np.unique(candidate_states))
        return (steps, np.unique(candidate_states))

    def test(self, testing_df, random_sample_size=100):
        """ Test the Model 

        Parameters
        ----------
        test_df : Dataframe
            The test dataframe
        """
        test = testing_df[["SubjectLine_ID","Sent_Day","Tenure_Group","Email_Domain","Age_Group","Gender","Type"]].values

        # Extract a random set from the test samples
        random_set = random.sample(range(len(test)), random_sample_size)

        # Initialize the metrics 
        metrics = Metrics()

        achieved_conversion_rates = []
        for i in random_set:
            subject = test[i][0]
            dow = test[i][1]
            tenure_group = test[i][2]
            email_domain = test[i][3]
            age_group = test[i][4]
            gender = test[i][5]
            type_val = test[i][6]
            
            state = (subject,dow, tenure_group, email_domain, age_group, gender, type_val)
            
            steps, candidate_states = self.get_candidate_states(state)

            predicted_conversion_rates = metrics.conversion_rates(self.email_campaign_df, candidate_states)
            
            achieved_conversion_rates += [np.mean(predicted_conversion_rates)]

        return metrics.run(achieved_conversion_rates)

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
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

    def predict(self, state):
        """ Predicts what states will give what conversions

        Parameters
        ----------
        state : object
            Initial state 
        """
         # Initialize the metrics 
        metrics = Metrics()

        steps, candidate_states = self.get_candidate_states(state)

        predicted_conversion_rates = metrics.conversion_rates(self.email_campaign_df, candidate_states)

        # Initialize the dictionaries
        subject_dict = {1: "Email Subject 1", 2: "Email Subject 2", 3: "Email Subject 3"}
        dow_dict = {0: "Sunday", 1: "Monday", 2: "Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday"}
        gender_dict = {0: 'Female', 1: 'Male'}
        type_dict = {0: 'Business', 1: 'Consumer'}
        domain_dict = {0: 'aol.com', 1: 'comcast.net', 2: 'gmail.com', 3: 'hotmail.com', 4: 'msn.com', 5: 'yahoo.com'}
        tenure_dict = {0:"< 5",1:"5 - 10", 2:"10 - 15", 3: "15 - 20", 4:"20 - 25", 5: "25 - 30",6:"> 30"}
        age_dict = {0:"< 20",1:"20 - 25", 2:"25 - 35", 3: "35 - 45", 4:"> 45"}

        print(domain_dict)

        audiences = list()
        i = 0
        for candidate_state in candidate_states:
            # Decode the state
            candidate = metrics.state_decoded(candidate_state)

            # Build the audience profile
            audience_profile = dict()
            print(candidate)
            audience_profile["Day of Week"] = dow_dict[candidate[1]]
            audience_profile["Tenure Group"] = tenure_dict[candidate[2]]
            audience_profile["Email Domain"] = domain_dict[candidate[3]]
            audience_profile["Age Group"] = age_dict[candidate[4]]
            audience_profile["Gender"] = gender_dict[candidate[5]]
            audience_profile["Customer Type"] = type_dict[candidate[6]]

            audience_dict = dict()
            audience_dict["audience-profile"] = audience_profile
            audience_dict["expected-conversions"] = predicted_conversion_rates[i]

            audiences.append(audience_dict)

            i += 1

        response_dict = dict()
        response_dict["Email Subject"] = subject_dict[state[0]]
        response_dict["Audience Permutations"] = audiences

        return response_dict
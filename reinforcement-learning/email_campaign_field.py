import numpy as np
import pandas as pd
from datetime import datetime, date 

class EmailCampaignField:
    def __init__(self, df, starting_state, states=None, target_conversion=0.1, email_sent_threshold=100,reward_factor=3):
        # Obtain the unique values for various attributes
        unique_subjects = np.sort(df['SubjectLine_ID'].unique())
        unique_weekday = np.sort(df['Sent_Day'].unique())
        unique_tenures = np.sort(df['Tenure_Group'].unique())
        unique_domains = np.sort(df['Email_Domain'].unique())
        unique_age_groups = np.sort(df['Age_Group'].unique()) 
        unique_genders = np.sort(df['Gender'].unique())
        unique_types = np.sort(df['Type'].unique()) 
        unique_response_results = np.sort(df['Response_Received'].unique())
       
        # Get attributes for the state space
        subject_ids = len(unique_subjects)
        days_of_week = len(unique_weekday)
        tenure_groups = len(unique_tenures)
        domains = len(unique_domains)
        age_groups = len(unique_age_groups)
        genders = len(unique_genders)
        types = len(unique_types)
        response_types = len(unique_response_results)
        
        # Create dictionary of keys for emails sent and received for a given subject
        self.emails_sent_by_subject = {}
        self.emails_responded_by_subject = {}
        for keys in unique_subjects:
            self.emails_sent_by_subject[keys] = 0
            self.emails_responded_by_subject[keys] = 0
        
        # Get the dimensions of the space
        self.subject_ids = subject_ids
        self.days_of_week = days_of_week
        self.tenure_groups = tenure_groups
        self.domains = domains
        self.age_groups = age_groups
        self.genders = genders
        self.types = types
        self.response_types = response_types
        
        # Build the state table
        if (states is None):
            self.build_state(df)
        else:
            self.states = states
        
        # Set the target state
        self.state = starting_state
        self.target_conversion = target_conversion
        self.reward_factor = reward_factor
        self.email_sent_threshold = email_sent_threshold

    def get_states(self):
        return self.states
    
    def build_state(self, df):
        # Initialize the state matrix
        emails_sent = np.zeros(self.get_number_of_states(),dtype=int)
        emails_responded = np.zeros(self.get_number_of_states(),dtype=int)

        print('Here we will build the state table')
        for index, row in df.iterrows():
            subject_id = row['SubjectLine_ID']
            day_of_week = row['Sent_Day']
            tenure_group = row['Tenure_Group']
            email_domain = row['Email_Domain']
            age_group = row['Age_Group']
            gender = row['Gender']
            type_attr = row['Type']
            response_type = row['Response_Received']

            # Encode the state
            subject_id_idx = subject_id - 1
            state = subject_id_idx*self.days_of_week*self.tenure_groups*self.domains*self.age_groups*self.genders*self.types
            state = state + day_of_week*self.tenure_groups*self.domains*self.age_groups*self.genders*self.types
            state = state + tenure_group*self.domains*self.age_groups*self.genders*self.types
            state = state + email_domain*self.age_groups*self.genders*self.types
            state = state + age_group*self.genders*self.types
            state = state + gender*self.types
            state = state + type_attr
            
            # Increment for this state to conversion as a tuple of (Total Responses Received, Total Emails Sent)
            current_emails_sent = emails_sent[state]
            current_emails_responded = emails_responded[state]

            emails_sent[state] = current_emails_sent + 1
            emails_responded[state] = current_emails_responded + response_type
        
        states = list()
        for i in range(0,len(emails_sent)):
            current_emails_sent = emails_sent[i]
            current_emails_responded = emails_responded[i]
            states.append((current_emails_responded,current_emails_sent))
        
        self.states = np.asarray(states)
        
    def get_number_of_states(self):
        return self.subject_ids*self.days_of_week*self.tenure_groups*self.domains*self.age_groups*self.genders*self.types
        
    def get_state(self):
        subject_id_idx = self.state[0] - 1
        state = subject_id_idx*self.days_of_week*self.tenure_groups*self.domains*self.age_groups*self.genders*self.types
        state = state + self.state[1]*self.tenure_groups*self.domains*self.age_groups*self.genders*self.types
        state = state + self.state[2]*self.domains*self.age_groups*self.genders*self.types
        state = state + self.state[3]*self.age_groups*self.genders*self.types
        state = state + self.state[4]*self.genders*self.types
        state = state + self.state[5]*self.types
        state = state + self.state[6]
        
        return state
    
    def get_state_outcome(self, target_state):
        subject_id_idx = target_state[0] - 1
        state = subject_id_idx*self.days_of_week*self.tenure_groups*self.domains*self.age_groups*self.genders*self.types
        state = state + target_state[1]*self.tenure_groups*self.domains*self.age_groups*self.genders*self.types
        state = state + target_state[2]*self.domains*self.age_groups*self.genders*self.types
        state = state + target_state[3]*self.age_groups*self.genders*self.types
        state = state + target_state[4]*self.genders*self.types
        state = state + target_state[5]*self.types
        state = state + target_state[6]
        
        return self.states[state]

    def get_reward_and_state_update(self, new_state):
        new_conversion = self.get_state_outcome(new_state)
        
        # Update conversions & get new conversion rate
        response_received, conversion_rate, conversion_goal_met = self.is_conversion_goal_met(new_state[0],new_conversion)
            
        # Compute State and reward
        if (conversion_goal_met):
            reward = self.reward_factor*10
        elif (response_received > 0):
            reward = 10
        else:
            reward = -1
        
        # Set the state
        self.state = new_state
        
        return (reward, conversion_goal_met)
        
    def make_action(self, action):
        state = self.state
        if (action == 6): # Next Subject Id
            subject_id = state[0] 
            next_subject_id = subject_id + 1
            if (next_subject_id > 3):
                next_subject_id = 1
            
            # Obtain the new state, get reward and update conversion rate
            new_state = (next_subject_id,state[1],state[2],state[3],state[4],state[5],state[6])
            return self.get_reward_and_state_update(new_state)

        elif (action == 0): # Next Day of the Week
            day_of_week = state[1] 
            next_dow = day_of_week + 1
            if (next_dow > 6):
                next_dow = 0
            
            # Obtain the new state, get reward and update conversion rate
            new_state = (state[0],next_dow,state[2],state[3],state[4],state[5],state[6])
            return self.get_reward_and_state_update(new_state)
        
        elif (action == 1): # Next Tenure Group
            tenure_group = state[2] 
            next_tg = tenure_group + 1
            if (next_tg > 6):
                next_tg = 0
            
            # Update conversions & get new conversion rate
            new_state = (state[0],state[1],next_tg,state[3],state[4],state[5],state[6])
            return self.get_reward_and_state_update(new_state)
        
        elif (action == 2): # Next Domain
            domain = state[3] 
            next_domain = domain + 1
            if (next_domain > 5):
                next_domain = 0
            
            # Update conversions & get new conversion rate
            new_state = (state[0],state[1],state[2],next_domain,state[4],state[5],state[6])
            return self.get_reward_and_state_update(new_state)
        
        elif (action == 3): # Age Group
            age_group = state[4] 
            next_ag = age_group + 1
            if (next_ag > 4):
                next_ag = 0
            
            # Update conversions & get new conversion rate
            new_state = (state[0],state[1],state[2],state[3],next_ag,state[5],state[6])
            return self.get_reward_and_state_update(new_state)
        
        elif (action == 4): # Gender
            gender = state[5] 
            
            next_gender = 1
            if (gender == 1):
                next_gender = 0
            
            # Update conversions & get new conversion rate
            new_state = (state[0],state[1],state[2],state[3],state[4],next_gender,state[6])
            return self.get_reward_and_state_update(new_state)

        elif (action == 5): # Type
            type_attr = state[6] 
            
            next_type = 1
            if (type_attr == 1):
                next_type = 0
            
            # Update conversions & get new conversion rate
            new_state = (state[0],state[1],state[2],state[3],state[4],state[5],next_type)
            return self.get_reward_and_state_update(new_state)

        else: # Penalize for wrong action
            return (-100, False)
   
    def is_conversion_goal_met(self, email_subject, conversion_stats):
        conversion_goal_met = False

        emails_responded = self.emails_responded_by_subject[email_subject] + conversion_stats[0]
        emails_sent =  self.emails_sent_by_subject[email_subject] + conversion_stats[1]

        conversion_rate = 0.0
        if (emails_sent > 0):
            conversion_rate = emails_responded / emails_sent
        
        #print(f"We have {emails_responded} responded emails out of {emails_sent} sent for subject {email_subject} with conversion rate {conversion_rate}")
        
        # If the conversion rate for the subject is above threshold we have met our goal for a subject
        if (conversion_rate > self.target_conversion and emails_sent > self.email_sent_threshold):
            conversion_goal_met = True
        
        # Set the subject wise conversion rate components
        self.emails_sent_by_subject[email_subject] = emails_sent
        self.emails_responded_by_subject[email_subject] = emails_responded
        
        return (conversion_stats[0], conversion_rate,conversion_goal_met)
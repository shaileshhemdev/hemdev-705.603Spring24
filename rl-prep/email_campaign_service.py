from flask import Flask
from flask import request, jsonify

import pandas as pd
import numpy as np

from email_campaign_field import EmailCampaignField

app = Flask(__name__)

"""
A class used to provides an REST API to get the next action to perform

As a part of service initialization we will load the q_table trained in our notebook
...

Methods
-------
detect_fraud()
    Will determine if the supplied transaction is fraudulent or not. To execute you use POST http://localhost:8786/detect-fraud

    Sample JSON as body below

    {
        "state": [1,0,0,0,0,0,0]
        "action": 0
    }

"""

@app.route('/get-next-action', methods=['POST'])
def get_next_action():
    """ Get next action given a state
        
    Parameters 
    ----------
    Encoded into Json with following mandatory attributes

    state : array
        Array of 7 state elements namely subject_id, 
    current_action : int
        Action to take where 0 = Subject, 1 = Day of Week, 2 = Tenure Group, 3 = Email Domain, 4 = Age Group, 5 = Gender, 6 = Type


    Sample JSON Below
    {
        "state": [1,0,0,0,0,0,0]
        "action": 0
    }
    """
    # Obtain request payload as a dictionary
    request_details = request.json

    # Obtain state 
    state = tuple(request_details["state"])
    action = request_details["action"]

    # Create Email campaign data
    email_campaign_data = EmailCampaignField(df,state, states)
    email_campaign_data.make_action(action)

    # Exploit to get the next best action
    next_state = q_table[email_campaign_data.get_state()]

    # Check if you want to avoid the same action again?
    #next_state[action] = -200.00
    # Return the next action
    next_action = np.argmax(next_state) 
    
    # Return the result as Json
    return jsonify({"next_action": next_action.item()})

if __name__ == "__main__":
    """ Initializes the Email Campaign Service Class

    """
    flaskPort = 8786

    # Load the Q Table
    q_table_df = pd.read_csv('q_table.csv')
    q_table = q_table_df[["Day of Week","Tenure Group","Email Domain","Age Group","Gender","Type"]].values
    print('Successfully obtained Q Table')

    # Load the transformed data 
    df = pd.read_csv('email_campaign_data.csv')
    print('Successfully obtained Campaign Data')

    # Initialize the Email Campaign Field
    starting_state = (1,0,0,0,0,0,0)
    email_campaign_field = EmailCampaignField(df,starting_state)
    states = email_campaign_field.get_states()

    # Now that all the setup has been done start the service
    print('Starting Server...')
    app.run(host = '0.0.0.0', port = flaskPort)


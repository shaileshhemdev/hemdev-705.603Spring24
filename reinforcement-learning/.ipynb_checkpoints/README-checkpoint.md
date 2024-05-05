# Email Campaign Reinforcement Learning

## Overview

This repository provides an initial notebook and service that trains on the Email Campaign Data and generates Q Table through Reinforcement Q Learning. [System Plan](SystemsPlan.md) outlines the thought process employed in firming scope & requirements, design methodology adopted and deployment & operations. It provides a simple yet complete end to end view of how to develop a machine learning system that can be leveraged in production 

## Local Development Environment used for building image

* Apple Mac M1 chip
* Sonoma 14.1.2
* Docker Desktop for Mac 4.26.1 (131620)
* Python 3.8
* Jupyter Notebook

## Running on Local

* You need git, python 3.8 and pip. See https://pip.pypa.io/en/stable/installation/

* It is recommended that you use Python virtual environments. See https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/

* Run pip3 install -r requirements.txt

* [Training Data](https://www.kaggle.com/datasets/aristotelisch/playground-mock-email-campaign) 

* Run python email_campaign_service.py. Pass the arguments for the data-folder and the training-data file. There are 2 ways to pass them namely system arguments like you see in the notebook example or via environment variables as you see in the docker example below. <b>NOTE</b>: At the moment not needed since both the <b>Preprocessed Campaign Data (email_campaign_data.csv)</b> and <b>Q Table (q_table.csv)</b> are committed in the repository. 

* Instead of above step, you can also use the notebook email_campaign_service_test_nb.ipynb. 

## Docker

### Pull Image

To pull an image use the following 

```
docker pull tomsriddle/rl-email-campaign:1.0

```

After pulling the image check that it is present using following

```
docker image ls

```

### Build Image from Local

```
docker buildx build -t "tomsriddle/rl-email-campaign:1.0" --load --platform linux/amd64,linux/arm64 .

```


### Run Image

To run the image use following

```
docker run -p 8788:8786 -v <Physical Host Folder>:/workspace/shared-data -e data-folder=/workspace/shared-data/email-campaign/ -e sent-emails-file=sent_emails.csv -e responded-emails-file=responded.csv -e customers-file=userbase.csv "tomsriddle/rl-email-campaign:1.0"

```
![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/EmailCampaignImageRun.png?raw=true)

## API Usage

Following APIs exposed 

### Get Next Action

This provides a mechanism to get the next best action to take given the state

```
POST http://localhost:8788/get-next-action

Request Body (all attributes are mandatory)
--------------------------------------------------------
{
        "state": [2,4,3,1,0,0,0],
        "action": 2
}

Response Body
--------------------------------------------------------

{
    "next_action": 4
}

```


![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/GetNextActionAPI.png?raw=true)

### Get Campaign Audience

This provides a mechanism to get audience permutations with high conversions

```
POST http://localhost:8788/campaign-audience?subjectId=2


Response Body
--------------------------------------------------------

{
    "campaign-audience": {
        "Audience Permutations": [
            {
                "audience-profile": {
                    "Age Group": "20 - 25",
                    "Customer Type": "Business",
                    "Day of Week": "Sunday",
                    "Email Domain": "aol.com",
                    "Gender": "Female",
                    "Tenure Group": "< 5"
                },
                "expected-conversions": 1.0
            },
            {
                "audience-profile": {
                    "Age Group": "25 - 35",
                    "Customer Type": "Business",
                    "Day of Week": "Sunday",
                    "Email Domain": "aol.com",
                    "Gender": "Female",
                    "Tenure Group": "< 5"
                },
                "expected-conversions": 0.0
            }
        ],
        "Email Subject": "Email Subject 2"
    }
}

```
![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/CampaignAudience.png?raw=true)


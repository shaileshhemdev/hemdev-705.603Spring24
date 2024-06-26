# Email Campaign Reinforcement Learning

## Overview

This repository provides an initial notebook and service that trains on the Email Campaign Data and generates Q Table through Reinforcement Q Learning. 

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
docker run -p 8788:8786 -v /Users/shaileshhemdev/ai/ai-enabledsystems/output:/workspace/shared-data -e data-folder=/workspace/shared-data/ -e "tomsriddle/rl-email-campaign:1.0"

```

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

## Notebook

Refer to [Email Campaigns Exploration](email-campaigns.ipynb). The notebook demonstrates the following

<ul>
    <li>
        <b>Preprocessing:</b>The preprocessing done to arrive at the final <b>email_campaign_data.csv</b> file
    </li>
    <li>
        <b>Email Campaign Field:</b>Encapsulation of the State Space, Actions and Reward Functions
    </li>
    <li>
        <b>Q Table:</b> Q Table Generation using above which generates the <b>q_table.csv</b> file
    </li>
</ul>


## Taxi environment (Other Notebooks)

A Q-learning implementation for OpenAIs Taxi-v3 environment. The notebook can be used as a tutorial for beginners in Reinforcement Learning.

## References

The following sources were used in constructing this notebook:

[1] OpenAI Gym. Taxi-v3 environment. The OpenAI Gym environment is available under the MIT License.  <br> https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py

[2] LearnDataSci. Reinforcement Q-Learning from Scratch in Python with OpenAI Gym. Taxi-v2 implementation. <br> https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

[3] Botforge. Save OpenAI Gym renders as GIFS. Public GitHub Gist. <br>
https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

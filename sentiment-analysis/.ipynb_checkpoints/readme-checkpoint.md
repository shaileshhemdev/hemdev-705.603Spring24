# Sentiment Analysis

## Overview

This repository provides a prototype implementation of an end to end service that can detect sentiments for a set of text reviews. [System Plan](SystemPlan.md) outlines the thought process employed in firming scope & requirements, design methodology adopted and deployment & operations. It provides a simple yet complete end to end view of how to develop a machine learning system that can be leveraged in production 

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

* [Training Data](https://livejohnshopkins.sharepoint.com/:u:/r/sites/Course_en_705_603_81_sp24-ejOViSujqfTlP/Shared%20Documents/General/amazon_movie_reviews.csv.zip?csf=1&web=1&e=VkKUPh) 

* Run python sentiment_analysis_service.py. Pass the arguments for the data-folder and the trainong-data file. There are 2 ways to pass them namely system arguments like you see in the notebook example or via environment variables as you see in the docker example below 

* Instead of above step, you can also use the notebook sentiment_analysis_test_nb.ipynb. 

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/FraudServiceTestingLocal.png?raw=true)

## Docker

### Pull Image


To check if you have the image already use this

```
docker image ls

```

### Build Image from Local

```
docker buildx build -t "tomsriddle/sentiment-analysis:1.0" --load --platform linux/amd64,linux/arm64 .

```


### Run Image

To run the image use following

```
docker run -p <host port>:8786 -v <host path>:/workspace/shared-data -e data-folder=/workspace/shared-data/ -e training-data-file=amazon_movie_reviews.csv "tomsriddle/sentiment-analysis:1.0" 

```

Note: See the volume mapping - this is needed for the data-folder where things like transformed_reviews.csv is stored. Similarly see the use of the 2 environment variables

#### Docker Image and Run Example

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/SentimentAnalysisImageBuild.png?raw=true)

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/sentiment-docker-run.png?raw=true)

## API Usage

Following APIs exposed 



### Get Sentiment 

This provides sentiments on an array of reviews sent

```
POST http://localhost:8788/get-sentiment

Request Body (all attributes are mandatory)
--------------------------------------------------------
{
    "reviews": [
        "The movie was awesome",
        "I got bored",
        "I hated it"
    ]
}

Response Body
--------------------------------------------------------

[
    2,
    1,
    0
]

```
![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/get-sentiment-api.png?raw=true)

## Model Evaluation

We have tried the following models

<ul>
    <li>
        <b>cardiffnlp/twitter-roberta-base-sentiment-latest:</b>
    </li>
   
</ul>


## Troubleshooting

* As a part of running from local or through docker image, we have to pass arguments for the folder where the  training data is. 
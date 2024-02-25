# Fraud Detection Model

## Overview

This repository provides a sample implementation of an end to end service that can predict if a transaction is fraudulent or not. [System Plan](SystemPlan.md) outlines the thought process employed in firming scope & requirements, design methodology adopted and deployment & operations. It provides a simple yet complete end to end view of how to develop a machine learning system that can be leveraged in production 

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

* [Training Data](https://jhu.instructure.com/courses/66217/files/9752451?wrap=1) 

* Run python fraud_service.py. Pass the arguments for the data-folder and the trainong-data file. There are 2 ways to pass them namely system arguments like you see in the notebook example or via environment variables as you see in the docker example below 

* Instead of above step, you can also use the notebook fraud_service_test_nb.ipynb. 

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/FraudServiceTestingLocal.png?raw=true)

## Docker

### Pull Image

To pull an image use the following 

```
docker pull tomsriddle/fraud-detection:1.0

```

After pulling the image check that it is present using following

```
docker image ls

```

### Build Image from Local

```
docker buildx build -t "tomsriddle/fraud-detection:1.0" --load --platform linux/amd64,linux/arm64 .

```


### Run Image

To run the image use following

```
docker run -p <host port>:8786 -v <host path>:/workspace/shared-data -e data-folder=/workspace/shared-data/ -e training-data-file=transactions-1.csv "tomsriddle/fraud-detection:1.0" 

```

Note: See the volume mapping - this is needed for the data-folder where things like transformed_data.csv is stored. Similarly see the use of the 2 environment variables

#### Docker Image and Run Example

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/FraudDetectionImageBuild.png?raw=true)

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/FraudDetectionImageRun.png?raw=true)

## API Usage

Following APIs exposed 

### Stats

This provides results (metrics) for testing data taken from the Training Data

```
GET http://localhost:8786/stats

Response Body
--------------------------------------------------------

{
    "Accuracy": 0.9975869064572656,
    "Average Precision Score": 0.5495803576746265,
    "Balanced Accuracy": 0.8121015145716799,
    "F1 Score": 0.7295171245310419,
    "Precision": 0.8766724840023269,
    "ROC AUC Score": 0.8121015145716798,
    "Recall": 0.6246632124352332,
    "Sensitivity": 0.6246632124352332,
    "Specificity": 0.9995398167081265
}

```

#### Stats Example

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/FraudDetectionStatsAPI.png?raw=true)

### Detect Fraud 

This provides prediction whether the given transaction is fraudulent or not. 

```
POST http://localhost:8788/detect-fraud

Request Body (all attributes are mandatory)
--------------------------------------------------------
{
    "trans_date_trans_time": "2019-01-01 00:00:18",
    "cc_num": "2703186189652095",
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "misc_net",
    "amt": 4.97,
    "first": "John",
    "last": "Doe",
    "sex": "F",
    "street": "57636 Russet Ln",
    "city": "South Lyon",
    "state": "MI",
    "zip": "48122",
    "lat": 36.079, 
    "long": -81.178,
    "city_pop": 2309,
    "job": "Psychologist, counselling",  
    "dob": "1988-03-09",      
    "trans_num": "0b242abb623afc578575680df30655b9",
    "unix_time" : 1325376018,
    "merch_lat": 36.011,
    "merch_long": -82.048
}

Response Body
--------------------------------------------------------

{
    "is_fraud": false
}

```

#### Valid Transaction Example

```
POST http://localhost:8788/detect-fraud

Request Body (all attributes are mandatory)
--------------------------------------------------------
{
    "trans_date_trans_time": "2019-01-02 01:06:37",
    "cc_num": "2703186189652095",
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "gas_transport",
    "amt": 2813.060,
    "first": "John",
    "last": "Doe",
    "sex": "F",
    "street": "57636 Russet Ln",
    "city": "South Lyon",
    "state": "MI",
    "zip": "48122",
    "lat": 29.440	, 
    "long": -99.727,
    "city_pop": 1595797,
    "job": "Soil scientist",
    "dob": "1960-10-28",
    "trans_num": "0b242abb623afc578575680df30655b9",
    "unix_time" : 1325468849,
    "merch_lat": 29.819,
    "merch_long": -99.143
}

Response Body
--------------------------------------------------------

{
    "is_fraud": true
}

```

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/FraudulentTransactionSample.png?raw=true)

## Model Evaluation

We have tried the following models

<ul>
    <li>
        <b>Random Forest:</b>This was by far the most robust with its ROC AUC and F1 Score staying in comparable ranges for Training, Validation and Testing Data. Due to this we eventually picked this
    </li>
    <li>
        <b>Gradient Boost:</b>Did extremely well on Training and Validation but not great on Test. So did not appear robust from POV of unseen data
    </li>
    <li>
        <b>ADA Boost:</b> Did not perform well with default parameters with close to 0 for F1
    </li>
    <li>
        <b>Naive Bayes:</b> Had low accuracy along with other metrics also being far lower
    </li>
</ul>

Results below between Random Forest,  Gradient Boost and Naive Bayes

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/ClassifierAccuracy.png?raw=true)

## Troubleshooting

* As a part of running from local or through docker image, we have to pass arguments for the folder where the fraud training data is. The service upon starting first runs the ETL Pipeline which will process the transaction data and create a transformed_data.csv file as shown below. If you don't see this then it means that the ETL Pipeline step has failed. 

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/TransformedFileCreated.png?raw=true)


* Note once the transformed file referenced above is created, it will not re-process the training data in transactions-1.csv. So if you want to repeat that process please delete the transformed_data.csv file


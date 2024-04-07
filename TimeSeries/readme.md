# Time Series Forecasting

## Overview

This repository provides a sample implementation of an end to end service that can predict total and fraudulent transactions given a date. 

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

* [Training Data](https://drive.google.com/file/d/1IKxNefL6pkBNBPdMpRJRy3OUxcXwzHHS/view?usp=sharing) 

* Run python time_series_service.py. Pass the arguments for the data-folder and the training-data file. There are 2 ways to pass them namely system arguments like you see in the notebook example or via environment variables as you see in the docker example below 

* Instead of above step, you can also use the notebook analysis/exploratory_data_analysis.ipynb. 

* In order to test on local you can use the notebook forecast_test_nb.ipynb as depicted below

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/ForecastingServiceLocal1.png?raw=true)
![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/ForecastingServiceLocal2.png?raw=true)

## Docker

### Pull Image

To pull an image use the following 

```
docker pull tomsriddle/time-series-forecasting:1.0

```

After pulling the image check that it is present using following

```
docker image ls

```

### Build Image from Local

```
docker buildx build -t "tomsriddle/time-series-forecasting:1.0" --load --platform linux/amd64,linux/arm64 .

```


### Run Image

To run the image use following

```
docker run -p <host port>:8786 -v <host path>:/workspace/shared-data -e data-folder=/workspace/shared-data/ -e training-data-file=CreditCardFraudFourYears.csv "tomsriddle/time-series-forecasting:1.0" 

```

Note: See the volume mapping - this is needed for the data-folder where things like transformed_data.csv is stored. Similarly see the use of the 2 environment variables

#### Docker Run Example

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/ForecastingServiceDockerRun.png?raw=true)

## API Usage

Following APIs exposed 

### Forecast  

This provides prediction whether the given transaction is fraudulent or not. 

```
POST http://localhost:8788/fraud-forecast

Request Body (all attributes are mandatory)
--------------------------------------------------------
{
    "forecast_date": "2023-11-28"
}

Response Body
--------------------------------------------------------

{
    "fraudulent_transactions": 33.09457837755995,
    "total_transactions": 16424.66484448641
}

```
![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/ForecastAPI.png?raw=true)

## Model Evaluation

We have tried the following models

<ul>
    <li>
        <b>SARIMAX:</b>Low RMSE for Total Transactions but higher for Fraudulent
    </li>
    <li>
        <b>Prophet:</b>Provided the same RMSE as SARIMAX but performed faster
    </li>
    <li>
        <b>LSTM (PyTorch):</b> Provided the lowest RMSE for Total Transactions
    </li>
</ul>


## Troubleshooting

* As a part of running from local or through docker image, we have to pass arguments for the folder where the  training data is. 


# Model Evaluation 

## Overview

This image provides a service that can predict the listing duration i.e. the time taken to sell a car from the time of listing given certain inputs. The service trains a Regression model on the basis of some trained data received and predicts the listing duration given some of these inputs

## Caveats

The implementation is example implementation and so the model is trained (fitted) upon the very first request sent. The trained model is maintained in memory which means stopping the container and starting it will need the model to be retrained. Also the data is fixed and very small

Real world implementations will have a lot more attributes, lot more data and the model saved in some sort of model repository at the very least. 

## Local Development Environment used for building image

* Apple Mac M1 chip
* Sonoma 14.1.2
* Docker Desktop for Mac 4.26.1 (131620)

## Features on the training data that were eliminated and the rationale

We have eliminated following features

<ul>
    <li>
        <b>engine_has_gas:</b> We get the same information from engine_type
    </li>
    <li>
        <b>engine_fuel:</b> This has values ['gasoline', 'gas', 'diesel', 'hybrid-petrol', 'hybrid-diesel', 'electric'] whereas engine_type is ['gasoline', 'diesel', 'electric']. The latter encompasses what would be needed in the previous one 
engine_capacity is only relevant for non electric vehicles and so can be omitted. 
    </li>
    <li>
        <b>feature_0 ... feature_9:</b>By eyeballing the records we see wide variance in duration even with same values for the features feature_0 to feature_9
    </li>
    <li>
        <b>location_region:</b>Wide variation within the same location
    </li>
    <li>
        <b>is_exchangeable:</b>Wide variation within the same location
    </li>
    <li>
        <b>model_name:</b>There are 1118 unique values and when group by manufacturer and model the standard deviation for the duration_listed is high
    </li>
    <li>
        <b>model_name:</b>There are 1118 unique values and when group by manufacturer and model the standard deviation for the duration_listed is high. The combination of transmission, body_type, engine_type, drivetrain give more richer information about the vehicle (e.g. under the same model Acura ZDX you get 3 body types and each one of them could influence the duration_listed more than the model in itself)
    </li>
    <li>
        <b>number_of_photos:</b> We removed this in the initial model
    </li>
    <li>
        <b>up_counter:</b> We removed this in the initial model
    </li>
</ul>


## Running on Local

* You need git, python 3.8 and pip. See https://pip.pypa.io/en/stable/installation/

* It is recommended that you use Python virtual environments. See https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/

* Run pip3 install -r requirements.txt

* Run python carfactors_service.py


## How to use the image using Docker

### Pull Image

To pull an image use the following 

```
docker pull tomsriddle/ml-carfactors:1.0

```

After pulling the image check that it is present using following

```
docker image ls

```

### Build Image from Local

```
docker buildx build -t "tomsriddle/ml-carfactors:1.0" --load --platform linux/amd64,linux/arm64 .

```


### Run Image

To run the image use following

```
docker run -p <host port>:8786 -v <host path>:/workspace/shared-data "tomsriddle/ml-carfactors:1.0" 

```

Note: See the volume mapping - this is needed for the saving of the image for the POST API

## API Usage

Following APIs exposed 

### Stats

```
http://localhost:8786/stats

```

#### Stats Example

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/Module4StatsCall.png?raw=true)

### Infer 

```
http://localhost:8786/infer?manufacturer=<Manufacturer>&transmission=<Transmission>&color=<Color>&engine=<EngineType>&drivetrain=<DriveTrain>&state=<State>&hasWarranty=<True|False>&odometer=<Odometer>&year=<YearofCar>&bodytype=<BodyType>&price=<PriceInUSD>

```

#### Infer Example

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/Module4InferCall.png?raw=true)





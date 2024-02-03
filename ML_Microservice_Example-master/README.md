# Example Machine Learning (ML) Microservice

## Overview

This image provides an example of a micro service that can predict whether a person with a given age or salary is likely to purchase a car or not. As a part of the inference, it runs a Classification algorithm using Random Forest Classifier to fit the model using some data that the service has and then uses this trained model

## Caveats

The implementation is example implementation and so the model is trained (fitted) upon the very first request sent. The trained model is maintained in memory which means stopping the container and starting it will need the model to be retrained. Also the data is fixed and very small

Real world implementations will have a lot more attributes, lot more data and the model saved in some sort of model repository at the very least. 

## Local Development Environment used for building image

* Apple Mac M1 chip
* Sonoma 14.1.2
* Docker Desktop for Mac 4.26.1 (131620)

## Changes Made over Initial Image

Following changes were made on the repository in order to push the image

* In requirements.txt instead of sklearn I specified scikit-learn. W/o this I encountered the following error * The 'sklearn' PyPI package is deprecated, use 'scikit-learn' rather than 'sklearn' for pip commands.* . For more details we can look at https://towardsdatascience.com/scikit-learn-vs-sklearn-6944b9dc1736

* During building using buildx while the build succeeded it did not publish any image to the local docker repository. *docker image ls* did not show the successfully built image. To resolve this I had to use the *--load* parameter and also remove one of the multi arch arguments. Details of this issue can be found here https://github.com/docker/buildx/issues/59. This solution finally worked https://github.com/docker/buildx/issues/59#issuecomment-1168619521

* In order to push a readme to Docker hub I also installed a CLI Plugin pushrm https://poweruser.blog/pushing-a-readme-file-to-docker-hub-68200bc4bf71. Using this executing *docker pushrm <REPOSITORY:IMAGE>* pushed the readme file

## How to use the image

### Pull Image

To pull an image use the following 

```
docker pull tomsriddle/ml-microservice:1.0

```

After pulling the image check that it is present using following

```
docker image ls

```

### Run Image

To run the image use following

```
docker run -p <host port>:8786 -v <host path>:/workspace/shared-data "tomsriddle/ml-microservice:1.0" 

```

Note: See the volume mapping - this is needed for the saving of the image for the POST API

## API Usage

Following APIs exposed 

### Stats

```
http://localhost:8786/stats

```

### Infer 

```
http://localhost:8786/infer?age=<age>&salary=<salary>

```

#### Bad Candidate Example

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/BadCandidate.png?raw=true)

#### Good Candidate Example

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/GoodCandidate.png?raw=true)

### Post Image 

Below shows the Post Body which uses Form as the Content-Type

![Local Repository Images](https://github.com/shaileshhemdev/public-images/blob/main/PostImage.png?raw=true)

In above, under Params we need have 2 params as shown below

![Local Repository Images](https://github.com/shaileshhemdev/public-images/blob/main/PostImageParams.png?raw=true)

Upon success, you should see the image saved as shown below

![Local Repository Images](https://github.com/shaileshhemdev/public-images/blob/main/SavedImage.png?raw=true)



# Graphical Degrading and Object Detection Assignment

## Overview

The repository contains an implementation of the OpenCV2 Deep Neural Network using YOLO model for object detection in images. It has the following key files 

<ul>
    <li>
        <b>Notebook (ObjectDetection.ipynb)</b>: In this notebook we demonstrate how certain reduction techniques on images can impact the object detection 
    </li>
    <li>
        <b>Python Class (object_detection.py)</b>: This class encapsulates the Deep Neural Network implementation to return the objects detected on a given image with the corresponding confidences and bounding boxes
    </li>
    <li>
        <b>Flask Service (object_detection_service.py)</b>: This provides a REST service to post any image to the service to obtain the objects detected along with their confidence
    </li>
</ul>
   
## Local Development Environment used for building image

* Apple Mac M1 chip
* Sonoma 14.1.2
* Docker Desktop for Mac 4.26.1 (131620)

## Running on Local

* You need git, python 3.8 and pip. See https://pip.pypa.io/en/stable/installation/

* It is recommended that you use Python virtual environments. See https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/

* Run pip3 install -r requirements.txt

* Run python object_detection_service.py


## How to use the image using Docker

### Pull Image

To pull an image use the following 

```
docker pull tomsriddle/cv-objectdetection:1.0

```

After pulling the image check that it is present using following

```
docker image ls

```

### Build Image from Local

```
docker buildx build -t "tomsriddle/cv-objectdetection:1.0" --load --platform linux/amd64,linux/arm64 .

```


### Run Image

To run the image use following

```
docker run -p <host port>:8786 -v <host path>:/workspace/shared-data -e data-folder=/workspace/shared-data/ "tomsriddle/cv-objectdetection:1.0" 

```

Note: See the volume mapping - this is needed for the saving of the image for the POST API. This volume should have the yolov3.cfg and yolov3.weights files as shown below

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/Module6YoloFilesInVolume.png?raw=true)

#### Docker Image and Run Example

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/Module6DockerRunCall.png?raw=true)


## API Usage

Following APIs exposed 

### Detect 

```
http://localhost:8786/detect

```
We need to submit form-data with image submitted as <b>imagefile</b> attribute 

#### Detect Example

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/Module6DetectCall.png?raw=true)

#### Detect Example in Docker

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/Module6DetectCallDocker.png?raw=true)

Notice how when you make the call the sent image is saved on your volume

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/Module6SentImagePersistedInVolume.png?raw=true)




# Exploratory Data Analysis 

## Overview

In this repository we are performing data analysis on the License Plate Recognition Dataset

## Dataset

You can obtain the dataset from here: https://jhu.instructure.com/courses/66217/files/10067213?wrap=1

## Notebook

You can open LicensePlateDetection.ipyng in your Jupyter notebook and run it

## Pre-requisites prior to executing the notebook

* You are using a Docker image for Jupyter Notebook
* Download the dataset provided in the Dataset section and unzip it on your local
* You will need python 3.8 and some key packages such as pandas, numpy, pylabel, opencv-python-headless and py
* You will need to install tesseract-ocr as outlined here on your Docker image https://tesseract-ocr.github.io/tessdoc/Installation.html (Note you might have a Mac or Windows but if you are using a docker image with jupyter then you need to install the Linux distribution for tesseract)
* There are also some plots that use matplotlib 

## Preparation prior to running notebook

* You have a "/workspace/shared-data/" folder mapped from your Docker image to physical directory that has the dataset downloaded
* You organize the unzipped contents under a folder such as "license-plates" in the above physical directory
* Organize the images under an "images" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/images")

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/LicensePlateImages.png?raw=true)

* Organize the XML annotations under an "annotations" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/annotations")

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/LicensePlateAnnotations.png?raw=true)

* Create a "cropped" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/cropped") for the cropped images

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/LicensePlateCroppedImages.png?raw=true)

* Create a "coco" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/coco") for the COCO Files

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/LicensePlateCocoFiles.png?raw=true)

* Create a "yolo" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/yolo") for the Yolo Files. Under "yolo" keep a "labels" sub folder

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/LicensePlateYoloFiles.png?raw=true)

## Conclusions

After running several samples I have drawn the following conclusions

* Tesseract does not do well with slanted images 
* Tesseract does not do well with double line images
* There are some parameters such as psm which has a range of values and some of them do well with some images but not with others
* Resizing an image by changing its angle a bit can improve detection of some images but make others suffer

## Future Work

Further research needs to be done on the following 

* I have chosen to make the image grayscale prior to passing it through Tesseract. Would like to research the impact of keeping the colors intact
* Would production grade systems use different parameter values based on the some preprocessing and inferences on the images

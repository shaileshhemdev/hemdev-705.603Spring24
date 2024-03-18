# Automated License Plate Recognition

## Overview

In this repository we are building License Plate Recognition Dataset. [System Plan](SystemsPlan.md) outlines the thought process employed in firming scope & requirements, design methodology adopted and deployment & operations. It provides a simple yet complete end to end view of how to develop a machine learning system that can be leveraged in production 

## Running on Local

```
python model.py udp://127.0.0.1:23000 /Users/shaileshhemdev/ai/ai-enabledsystems/workspace/license-plates/ prediction1/ original/ cropped/ models/ 2 1

```

## Pre-requisites prior to executing the notebook

* You are using a Docker image for Jupyter Notebook  
* Download the dataset provided in the Dataset section and unzip it on your local
* You will need python 3.8 and some key packages such as pandas, numpy, pylabel, opencv-python-headless and py
* You will need to install tesseract-ocr as outlined here on your Docker image https://tesseract-ocr.github.io/tessdoc/Installation.html (Note you might have a Mac or Windows but if you are using a docker image with jupyter then you need to install the Linux distribution for tesseract)

```
 apt-get update -qq && apt-get install tesseract-ocr -y
```

* On your docker image that has the notebook install ffmpeg 

```
 apt-get update -qq && apt-get install ffmpeg -y
```
* There are also some plots that use matplotlib 

## Notebooks

There are several notebook files and their purpose is outlined below

<ul>
    <li>
        <b>LicensePlateDetection.ipynb</b>: This notebook serves as a simple and quick wrapper to execute the code namely the Pipeline (data_pipeline.py), Model (model.py) and Metrics (metrics.py)
    </li>
    <li>
        <b>analysis/exploratory_data_analysis.ipynb</b>: This notebook serves as the detailed analysis of how we arrived at the design 
    </li>
    <li>
        <b>analysis/model_performance.ipynb</b>: This notebook serves as a performance comparison of the 2 Yolo models considered
    </li>
</ul>

To run, tou can open  in your Jupyter notebook and run it

## Components

Following are key components

<ul>
    <li>
        <b>data_pipeline.py</b>: This has a Pipeline class that will extract images from video streams, transform them into cropped images for left and right lanes and then load the resized images (608, 608) to work with YOLO models
    </li>
    <li>
        <b>dataset.py</b>: This has a Object_Detection_Dataset class that organizes the images into training, testing and validation datasets
    </li>
    <li>
        <b>model.py</b>: This has a Object_Detection_Model class that enables us to test and predict from a set of images what potential license plates there are
    </li>
    <li>
        <b>metrics.py</b>: This has a Metrics class that enables us to run and generate metrics relevant for this project. We have made use of Accuracy, Precision and Levinshtein Distance the latter is helpful when we don't get an exact match between predicted and actual license plate numbers
    </li>
    <li>
        <b>deployment_udp_client.py</b>: This class reads from video stream and saves read bytes as a JPEG image
    </li>
</ul>

To run, tou can open  in your Jupyter notebook and run it


## Preparation prior to running notebook

* You have a "/workspace/shared-data/" folder mapped from your Docker image to physical directory that has the dataset downloaded
* You create a folder "license-plates" in the above physical directory. This is to organize all outputs under this
* You create a folder "models" under above directory and keep the YOLO files (.CFG and .weights)
* Create "original" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/original"). The images from video streams will be saved here
* Create "cropped" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/cropped"). The preprocessing cropped images will be stored here
* Create "prediction" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/prediction1"). This is helpful when you are running the image for making a prediction

## Conclusions

After running several samples I have drawn the following conclusions

* The runs.txt files in reports sub folder show some of the results. 
* Tesseract does not do well with slanted images 
* Tesseract does not do well with double line images
* There are some parameters such as psm which has a range of values and some of them do well with some images but not with others
* Resizing an image by changing its angle a bit can improve detection of some images but make others suffer

## Future Work

Further research needs to be done on the following 

* I have chosen to make the image grayscale prior to passing it through Tesseract. Would like to research the impact of keeping the colors intact
* Would production grade systems use different parameter values based on the some preprocessing and inferences on the images

## VOC Pascal, Coco and Yolo Files (Not needed for this system)

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/LicensePlateImages.png?raw=true)

* Organize the XML annotations under an "annotations" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/annotations")

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/LicensePlateAnnotations.png?raw=true)

* Create a "cropped" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/cropped") for the cropped images

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/LicensePlateCroppedImages.png?raw=true)

* Create a "coco" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/coco") for the COCO Files

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/LicensePlateCocoFiles.png?raw=true)

* Create a "yolo" sub-folder under "license-plates" (E.g. /workspace/shared-data/license-plates/yolo") for the Yolo Files. Under "yolo" keep a "labels" sub folder

![Image Not Showing](https://github.com/shaileshhemdev/public-images/blob/main/LicensePlateYoloFiles.png?raw=true)


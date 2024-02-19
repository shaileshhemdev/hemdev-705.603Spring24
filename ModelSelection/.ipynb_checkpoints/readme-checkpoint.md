
# Model Selection

## Overview

There are different types of Models available within Machine Learning and one has to first determine which model(s) are relevant to the problem at hand. [ModelTypes.pdf](ModelTypes.pdf) shows the various types of models available and [ModelSelect.pdf](ModelSelect.pdf) shows how one can go about selecting a model. In this repository, we use a problem of SpeedDating to demonstrate the different models that can be applied to the same Machine Learning problem

Additionally there is the CarFactors problem where we demonstrate how to write a model and convert it into a micro service which allows exposing the trained model for use in predicting actual car sale time. 

## Local Development Environment used for building image

* Apple Mac M1 chip
* Sonoma 14.1.2
* Docker Desktop for Mac 4.26.1 (131620)
* Python 3.8
* Jupyter Notebook

## Speed Dating

The key file in this repository is the [Model Selection Exercise](ModelSelectionExercise.ipynb) notebook. You need Jupyter installed or an image of Jupyter Labs available to run this notebook along with Python 3.8

### Models Covered

In this repository, we have outlined 2 Models for the Speed Dating namely <b>Dimensionality Reduction</b> and <b>Supervised Learning for Classification</b>

#### Dimensionality Reduction

The Speed Dating data consists of 195 columns i.e. 194 features many of which are hard to understand. Data for many of these attributes is collected via surveys and hence is sparse making it hard to impute. It is also not clear what features are correlated. Thus till we reduce these dimensions methodically to keep the ones that have the most impact we run the risk of creating models that don't provide a good prediction. We employ <b>Principal Component Analysis</b> that creates new components and explains the variances between these components so that we can pick the top ones that account for most of the variance thereby giving us the new features that provide the best prediction. 

While PCA is great in significantly reducing dimensions but yet providing the needed dimensions for supervised learning, the biggest challenge is that these dimensions are new ones computed from the original features and hence hard / impossible to undertand 

#### Supervised Learning

After reducing the dimensions, there is a classic supervised learning classification problem of predicting a match that can potentially help in making the entire speed dating process more effective by matching participants that are most likely to go on dates after the event  

## Projected Car Sale Time Inference

To understand how to expose a model as a service proceed to [Car Factors](CarFactors/README.md). 

### Models Covered

For Car Sales Time Prediction the repository demonstrates using <b>Random Forest Classifier</b> from sklearn

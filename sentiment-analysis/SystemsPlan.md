# Sentiment Analysis System Design

## MOTIVATION (Why's)

### Why are we solving this problem?

The rating system has a fixed set of ratings going from 1 star to 5 star. Customers often make explicit choices that are not backed by what they express when asked to articulate their experience. Gleaning the overall sentiment from what the qualitative review enables us to get a more consistent sentiment associated with the review. 

#### Problem Statement

The key problem we are trying to solve is to get the sentiment from the text review using additional attributes available that might influence or boost the sentiment

#### Value Proposition

By understanding the sentiment, we can build a more accurate recommendation system as well as enable our customers to rely more on the reviews when making a decision

### Why is our solution a viable one?

At the very least, we need to determine whether the sentiment associated with the review is Positive, Neutral or Negative. If we are able to go further such as making it Loved It, Liked It, OK, Did not like it and Do No Recommend we would be providing a more granular view to others 

<ol>
    <li>
        <b>Is Machine Learning the right fit?: </b> Machine learning is needed here since Doing it manually would take a lot of time for humans to parse through content. Humans are also like to put their bias into the analysis and given the arbirary nature of text we cannot codify it using deterministic rules
    </li>
    <li>
        <b>Can we tolerate mistakes and a degree of risk?: </b> Since the purpose of sentiment analysis is to detect the sentiment which at best can drive other customers from making their decision on the product or service for which the sentiment was expressed we can tolerate a good deal of risk here. 
    </li>
    <li>
        <b>Is this solution (with ML) feasible? : </b> Natural Language Processing as well as Large Language Models enable us to take arbitrary text and glean the sentiment from it. Thus we can send a variety and large volume of text and determine what the provider of the content was likely feeling (aka sentiment). 
    </li>
</ol>

## REQUIREMENTS (What's)

### SCOPE

#### What are our goals?

Following are the goals of the system

<ol>
    <li>
        <b>Organizational: </b> Identify the sentiment associated with a text with the potential value being Positive, Negative or Neutral
    </li>
    <li>
        <b>System:</b>Efficient use of storage, low cost and great customer experience. Text to be reviewed is capped
    </li>
    <li>
        <b>User:</b>Users are able to provide additional contextual parameters to aid the sentiment detection
    </li>
    <li>
        <b>Model:</b>Model should have high accuracy
    </li>
</ol>

#### What are the success criteria?

The above goals can be expressed in the following success criteria

<ol>
    <li>
        <b>Metric Design: </b>The design of the metrics should consider the following  
        <ol>
            <li>Model should be able to handle large amounts of text</li>
            <li>Metrics should enable identifying the sentiment accurately at least 90% of the time</li>
            <li>It should have a very high precision (>90%) and high recall (>70%)</li>
        </ol>
    </li>
    <li>
        <b>Metrics Evaluation: </b>The evaluation of the metrics need to factor the following
        <ol>
            <li>System should be able to retrain the model at prescribed intervals</li>
            <li>System should be able to replace the models used for training and predicting</li>
            <li>System should be able to monitor the performance of the model by running test transactions</li>
        </ol>
    </li>
</ol>

### REQUIREMENTS

#### What are our (system) Assumptions?

Following are the assumptions made by the system

<ol>
    <li>All data needed for the system is collected and accessible by the system</li>
    <li>Changes in data such as additional attributes or value enumerations or data types are communicated in advance to the system in order to adapt and adjust before these changes are put into production</li>
    <li>Sufficient guard rails are present upstream to foster trust in the data and ensuring data is accurate</li>
    <li>Sufficient validations are in place upstream to avoid incomplete or missing data</li>
    <li>Business Domain Experts are available to provide suggestions and recommnedations on how to cleanse data including imputations for missing data</li>
    <li>A large amount of historical labelled (ground truth) data is available to initially train the models and this data is certified by the domain experts</li>
</ol>


#### What are our (system) Requirements?

Based on the goals we come up with the following requirements 

<ol>
    <li>
        <b>Functional: </b>Here are the functional requirements of the system
        <ol>
            <li>System should predict if a sentiment in a text is Positive, Negative or Neutral</li>
            <li>System should enable the user (customer) to provide additional context</li>
            <li>System should allow authorized users (business users of financial institution) to test random text to measure the performance of the system</li>
           <li>System should allow authorized users (machine learning engineers) to test different models</li>
           <li>System should allow authorized users (machine learning engineers) to retrain models on existing data</li>
           <li>System should at least 70% Recall</li>
           <li>System should at least 90% Precision</li>
           <li>System should at least 90% Accuracy</li>
        </ol>
    </li>
    <li>
        <b>Non Functional Requirements: </b>Here are the non-functional requirements of the system
        <ol>
           <li>System should allow authorized users (machine learning engineers) to pre process data to be fed into a model for training to cover for situations when the logic of imputation has changed or upstream bugs are discovered</li>
            <li>System should allow authorized users (machine learning engineers) to change thresholds for the model and verify if the model meets the thresholds specified on validation data</li>
        </ol>
    </li>
</ol>

### RISK & UNCERTAINTIES

#### What are the possible harms?

Here are some of the harms that we see 

<ol>
    <li>Poor Model Quality leads to misclassification</li>
    <li>Poor Model Quality leads to users not trusting the sentiment and thereby the review system</li>
    <li>Some reviews are more vulnerable to inaccurate sentiment detection</li>
    <li>Sudden dips and surges in the metrics impacting the overall performance of the model</li>
    <li>Model performance degrades over time</li>
</ol>


#### What are the causes of mistakes?

Here are some of the causes that lead to mistakes

<ol>
    <li>Data Systems are breached where the data used for training is corrupted or manipulated</li>
    <li>Bugs in upstream systems lead to inaccurate or incomplete data</li>
    <li>Model performs well during training and even validation but not in actual field</li>
    <li>Model does not adapt to changing shape and quality of data</li>
</ol>

## IMPLEMENTATION (How's)

### DEVELOPMENT

#### Methodology

Our methodology involves the following Software Development Lifecycle processes 

<ol>
    <li>
        <b>Data Analysis:</b>In this phase sample datasets or subsets of sample data would be loaded into local development workspaces such as Jupyter notebooks to help understand the data. We will build visualizations and the purpose here would be to get some clues or intuition on what potentially could be affecting the end result. We have done that for this dataset in a notebook under <b>analysis/exploratory_data_analysis.ipynb</b>. In this notebook we are using direct pandas and exploring the data to get some intuition on what could be good candidate features, what cleansing and normalizing we need to do on the data and what models might work out
    </li>
    <li>
        <b>Dataset Processing:</b>We have leveraged a module <b>dataset.py</b> with a class <b>Text_Pipeline</b> which has functions to prepare the text. It removes whitespace, punctuation, stop words, converts numbers 
    </li>
    <li>
        <b>Data Partitioning:</b>We have leveraged a module <b>data_pipeline.py</b> with a class <b>Sentiment_Analysis_Dataset</b> which employs a K-Fold (default folds 5) to create training and testing data subsets. We use these subsets to provide the training and testing datasets to consumers of this class. There is also a get_validation_dataset method that applies a split from the training data as validation data. 
    </li>
    <li>
        <b>Model Training and Metrics</b>We have leveraged a module <b>metrics.py</b> with a class <b>Metrics</b> to return metrics on testing data as well as to generate a report. There is a notebook under <b>analysis/model_performance.ipynb</b> that leverages all these classes and simulates metrics through various classifiers we have tried. We have used multiple metrics namely Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, Recall, F1 Score, ROC AUC Score and Average Precision Score. Of these <b>ROC AUC Score</b>, <b> F1 Score </b> and <b> Balanced Accuracy </b> is what we have deemed relevant for this dataset. Accuracy is ruled out since given the heavy imbalance of the class labels, we can get high accuracy even when the model is performing poorly (see the results for Ada Boosting as an example). 
    </li>
    <li>
        <b>Model Selection Analysis:</b>We have tried 1 classifiers Transformer with <b>cardiffnlp/twitter-roberta-base-sentiment-latest</b> model
    </li>
    <li>
        <b>Deployment Strategy:</b>We have packaged the interface into a Flask app in the module <b>sentiment_analysis_service.py</b>. This app exposes a REST API interface and performs the cleansing, feature selection and training of the model during the start up phase. It exposes 2 end points one for '/stats' that sends testing split using the Sentiment Analysis Dataset module to get the statistics which we have discussed in the Metrics. The second end point is '/get-sentiment' that takes in a payload for a text reviews and returns sentiments. We also have a <b>sentiment_analysis_service_test_nb.ipynb</b> Notebook that starts this service to test it on local
    </li>
</ol>

#### High-level System Design

Our overall system design involves the following key components

<ol>
    <li>
        <b>Pipeline:</b>We will build a Machine Learning pipeline that has several steps and data can flow into and out from these steps where each step performs specific tasks such as pre-processing or training
    </li>
    <li>
        <b>Data Pre-processing:</b>This component will prepare the data so that it is usable for any learning process. This involves imputing missing values, performing transformations such as removing stop words. 
    </li>
    <li>
        <b>Feature Engineering:</b>Currently we are only using the text but overall goals are to look at additional models where we can use additional data as context to influence the sentiment
    </li>
    <li>
        <b>Dataset:</b>This component will provide relevant portions of the training, validation and test datasets that can be leveraged by other modules
    </li>
    <li>
        <b>Training:</b>This component will train the model and perform analysis such as cross validation or multi fold analysis to ensure that will enable choosing the best hyperparameters for the model and ensure that the model is robust across all the combinations of the data 
    </li>
    <li>
        <b>Testing:</b>This component will enable testing the trained models both on previously seen data such as a subset from the original dataset but also from never seen before data. 
    </li>
    <li>
        <b>Metrics:</b>This component will provide the needed metrics on the model for the dataset provided persisting it for traceability
    </li>
</ol>

#### Development Workflow

As described in the Methodology section we will follow the Scoping --> System Design --> Data Analysis --> Modeling --> Development --> Deployment --> Operations workflow

### POLICY


#### Human-Machine Interfaces

We have assumed that following upstream human machine interactions would exist

<ol>
    <li>
        <b>Customer Fraud Engagement Feedback:</b>We have assumed that an upstream system will log the actual rating class when the customer is asked to provide a review. This helps us gauge accuracy 
    </li>
    <li>
        <b>New Ground Truth Data:</b>We expect an upstream system to periodically send us new training data on the basis of the latest set of transactions joined with the feedback received from customer. Thus the service provider team comprising of ML engineers, Data Scientists and Software engineers would analyze this data and find ways to improve the model with new classifiers, some more transformations and releasing it to production to measure performance. 
    </li>
</ol>

#### Regulations

We have researched for following resources that speak to what a financial institution needs to be aware of and in compliance with 

<ol>
    <li>
        <b>OCC:</b>https://www.occ.treas.gov/news-issuances/bulletins/2019/bulletin-2019-37.html
    </li>
    <li>
        <b>CFPB:</b>https://www.consumerfinance.gov/rules-policy/regulations/1005/6/
    </li>
    <li>
        <b>FDIC:</b>https://www.fdic.gov/resources/supervision-and-examinations/examination-policies-manual/section9-1.pdf
    </li>
    <li>
        <b>DOJ:</b>https://www.justice.gov/archives/jm/criminal-resource-manual-958-fraud-affecting-financial-institution
    </li>
</ol>

### OPERATIONS

#### Continuous Deployment

We will need to follow up with the following upgrades to make this a production grade system

<ol>
    <li>
        <b>CI/CD Integration:</b>Making this part of CI/CD tooling such as Jenkins or Code Deploy so that we are building the code with each commit, checking for code quality (Sonar), measuring test coverage, building and storing versions and finally deploying it in a non invasive manner using Blue Green or Canary style deployments so that we can 'Launch Dark' and get early feedback before the changes are truly live. 
    </li>
    <li>
        <b>Automated Tests:</b>Need suite of tests built for the software components as well as for the Model performance using synthetic data generation 
    </li>
</ol>


#### Post-deployment Monitoring

We will need to set up some monitors to get real time feedback on the following 

<ol>
    <li>
        <b>Data Quality:</b>Ensuring we are getting valid values for all attributes 
    </li>
    <li>
        <b>Model Performance:</b>Feedback from the user engagement to tell us real time if the prediction was accurate or not
    </li>
    <li>
        <b>Re-training:</b>As we collect new transactions we want the service to periodically re-train on the fresh data with a mix between latest and historical
    </li>
</ol>

#### Maintenance

Classes are loosely coupled that allows us to change the ETL Pipeline from the Dataset Partitioning from the Model itself. This allows us to make changes to one of them without affecting the other. 

#### Quality Assurance

We have made use of notebooks to measure the quality 

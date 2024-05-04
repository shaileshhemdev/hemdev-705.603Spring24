# Email Campaign System Design

## MOTIVATION (Why's)

### Why are we solving this problem?

Marketing Teams want to send campaign emails to potential customers with an intention that they evoke interest leading ultimately to the purchase of a product or a service that they are offering. From a campaign stand point, a valuable metric is whether a potential customer read the email and responded back to the service provider with questions, intent to purchase, etc which serves as a concrete metric that the campaign worked for that customer. This response to an email is referred to as conversion. Ultimately if conversion is low then the marketing cost is wasteful and does not yield the outcomes that the service provider is aiming for. Thus being able to design the campaign with target conversion rates and then meeting those rates is essential to justify the costs involved in marketing. 

#### Problem Statement

Given customer demographic data and past history of conversions, we want to be able to identify the  audience of an email campaign involving  email content uniquely identified by a Email Subject Id such that it meets our target conversion rate. Conversion Rate is defined as the ratio of Emails Responded : Emails Sent. As a campaign manager, once I have designed the email content I want to be able to identify the exact customer population to target this to by simulating potential conversion rates for different permutations of customer demographics

#### Value Proposition

By targeting email campaigns to specific customers leveraging previous campaign data, we have a better chance of reaching conversion rates

### Why is our solution a viable one?

We have a history of 2.5M emails (2476354 to be precise) across 3 Email Subjects, 7 potential days of the week, 5 age groups, 6 tenure groups, 6 email domains, 2 genders and 2 types

<ol>
    <li>
        <b>Is Machine Learning the right fit?: </b> Yes. There are overall <b>17640 permutations</b> = 3 (Subject Ids) x 7 (Days of Week) x 7 (Tenure Groups) x 6 (Email Domains) x 5 (Age Groups) x 2 (Gender) x 2 (Type). Going through these manually is time consuming and so with machine learning we can arrive at our target permutation faster with a series of moves
    </li>
    <li>
        <b>Can we tolerate mistakes and a degree of risk?: </b> The main risk is that if we cannot meet the target conversion rate with machine learning then we have incurred additional cost of implementing a technological solution (infrastructure cost, software engineering cost, etc) to add to the marketing cost all of which now is not giving us the benefits. There is also the risk of our email domain being considered spam due to multiple customers marking it as such preventing any future campaigns from being successful. Thus while we can tolerate some mistakes, there is a high risk of losing leverage in the future if we get our campaign targets wrong
    </li>
    <li>
        <b>Is this solution (with ML) feasible? : </b> The core problem is to optimally get to the permutation that can give us the highest conversion rate. Reinforcement Learning is one of the techniques we can use to obtain this
    </li>
</ol>

## REQUIREMENTS (What's)

### SCOPE

#### What are our goals?

Following are the goals of the system

<ol>
    <li>
        <b>Organizational: </b> Identify the target audience in optimal time that can be repeated for multiple campaigns
    </li>
    <li>
        <b>System:</b>Efficient use of compute and resources to arrive at the target audience
    </li>
    <li>
        <b>User:</b>Users (Campaign Managers) are able leverage an interface to try different permutations and get the likely conversion rate for that permutation
    </li>
    <li>
        <b>Model:</b>Model should have high accuracy between projected conversion and actual conversion
    </li>
</ol>

#### What are the success criteria?

The above goals can be expressed in the following success criteria

<ol>
    <li>
        <b>Metric Design: </b>The design of the metrics should consider the following  
        <ol>
            <li>Model should be able to handle highly unbalanced data since the percentage of Fraud is less than 1% of total transactions and hence we need appropriate metrics that provide a high degree of accuracy for such data</li>
            <li>Metrics should enable identifying a fraudulent transaction 90% of the time and enable not exceeding the number of times a valid transaction is identified as Fraud by 30%</li>
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
            <li>System should predict if a given transaction is fraudulent or not</li>
            <li>System should enable the user (customer) to indicate if the prediction is accurate or not by approving or declining a fraudulent transaction</li>
            <li>System should allow authorized users (business users of financial institution) to test random transactions to measure the performance of the system</li>
           <li>System should allow authorized users (machine learning engineers) to test different models</li>
           <li>System should allow authorized users (machine learning engineers) to retrain models on existing data</li>
           <li>System should at least 70% Recall</li>
           <li>System should at least 90% Precision</li>
           <li>System should at least 0.8% ROC AUC Score</li>
           <li>System should at least 0.7% F1 Score</li>
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
    <li>Poor Model Quality leads to high number of fraudulent transactions going undetected leading to large scale monetary risk for the financial insitution as well as loss of customers</li>
    <li>Poor Model Quality leads to high number of valid transactions predicted as fraud leading to large scale customer dissatisfaction</li>
    <li>Customers in certain age groups or races or locations are more vulnerable to fraud going undetected versus the others leading to a bias perception</li>
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
        <b>Dataset Processing:</b>We have leveraged a module <b>dataset.py</b> with a class <b>ETL_Pipeline</b> which follows a Extract, Transform and Load cycle. During extract it reads training data from supplied <b>Fraud Transactions Training Data (transactions-1.csv)</b>. In the <b>extract</b> phase (method) we load the raw data and create a dataframe. In the <b>transform</b> phase (method) we drop columns we don't need but prior to that generate some higher order derived features such as Age, Transaction Day of the Week, Transaction Month, Transaction Time of the Day Segment, Distance between Customer and Merchant's coordinates. We also encode Category into more higher order categories such as Shopping, Entertainment, Home and Misc as well as identify if the transaction is a Internet based or Physical Store based transaction since our analysis shows these have some correlation to the target. This derivation of higher order features enables us to drop a lot of attributes such as customer address components, coordinates for customer and merchant in terms of lat/long, customer name, dob, etc. Finally we apply scaling to all numeric data and encode attributes such as Job Code. Our final features are (Label Encoded), Category (One Hot Encoded into broader categories namely Entertainment,Home,Misc,Shopping) ,Transaction Day of Week (Ordinal Encoded),Transaction Month (Ordinal Encoded) ,Transaction Part of Day (Ordinal Encoded), Amount (Min Max Scaled) ,City Population (Min Max Scaled),Age (Min Max Scaled),Distance from Merchant (Min Max Scaled)
    </li>
    <li>
        <b>Data Partitioning:</b>We have leveraged a module <b>data_pipeline.py</b> with a class <b>Fraud_Dataset</b> which employs a Stratified K-Fold (default folds 5) to create training and testing data subsets. We use these subsets to provide the training and testing datasets to consumers of this class. There is also a get_validation_dataset method that applies a split from the training data as validation data. 
    </li>
    <li>
        <b>Model Training and Metrics</b>We have leveraged a module <b>metrics.py</b> with a class <b>Metrics</b> to return metrics on testing data as well as to generate a report. There is a notebook under <b>analysis/model_performance.ipynb</b> that leverages all these classes and simulates metrics through various classifiers we have tried. We have used multiple metrics namely Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, Recall, F1 Score, ROC AUC Score and Average Precision Score. Of these <b>ROC AUC Score</b>, <b> F1 Score </b> and <b> Balanced Accuracy </b> is what we have deemed relevant for this dataset. Accuracy is ruled out since given the heavy imbalance of the class labels, we can get high accuracy even when the model is performing poorly (see the results for Ada Boosting as an example). 
    </li>
    <li>
        <b>Model Selection Analysis:</b>We have tried 3 classifiers Random Forest, Gradiant Boosting and Ada Boosting. Of these we found <b>Random Forest</b> to have the best metrics across different folds. There is a notebook under <b>analysis/model_performance.ipynb</b> that leverages all these classes and simulates metrics through various classifiers we have tried. 
    </li>
    <li>
        <b>Deployment Strategy:</b>We have packaged the interface into a Flask app in the module <b>fraud_service.py</b>. This app exposes a REST API interface and performs the cleansing, feature selection and training of the model during the start up phase. It exposes 2 end points one for '/stats' that sends testing split using the Fraud_Dataset module to get the statistics which we have discussed in the Metrics. The second end point is '/detect-fraud' that takes in a payload for a transaction and returns a prediction if it is fraudulent or not. We also have a <b>fraud_service_test_nb.ipynb</b> Notebook that starts this service to test it on local
    </li>
</ol>

#### High-level System Design

Our overall system design involves the following key components

<ol>
    <li>
        <b>Pipeline:</b>We will build a Machine Learning pipeline that has several steps and data can flow into and out from these steps where each step performs specific tasks such as pre-processing or training
    </li>
    <li>
        <b>Data Pre-processing:</b>This component will prepare the data so that it is usable for any learning process. This involves imputing missing values, scaling and normalizing the data, performing transformations such as address normalizations or case sensitivities or encoding or labelling data. The final transformed data will flow through the Data Processing Pipeline to the next stage often saving the transformed files somewhere for efficient usages by the next stages in the pipeline
    </li>
    <li>
        <b>Feature Engineering:</b>This component will analyze all the features and drop features that are not necessary. Additionally it might perform some higher order transformations that generate new features that are permutations and combinations of the original input features to create the final set of features to be used for training
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
        <b>Engagement based on Fraud Prediction:</b>Our service predicts whether a transaction is fraud is not. System consuming this information along with upstream systems would need to engage the user to confirm if it is a fraud prior to approving or declining the transaction. 
    </li>
    <li>
        <b>Customer Fraud Engagement Feedback:</b>We have assumed that an upstream system will log the actual fraud class when the customer is asked to confirm if a transaction is fraudulent or not. Similarly if a fraudulent transaction is not predicted but the customer engages the institution after the incident, the same would be captured. 
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

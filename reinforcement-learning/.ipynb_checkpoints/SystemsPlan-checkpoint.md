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
        <b>Organizational: </b> Enable campaign managers to identify the target audience for a given email content (uniquely identified by Subject Id)
    </li>
    <li>
        <b>System:</b>Efficient use of compute and resources to arrive at the target audience
    </li>
    <li>
        <b>User:</b>Users (Campaign Managers) are able leverage an interface to try different permutations and get the likely conversion rate for that permutation
    </li>
    <li>
        <b>Model:</b>Model should have ensure actual conversion is greater than desired conversion. 
    </li>
</ol>

#### What are the success criteria?

The above goals can be expressed in the following success criteria

<ol>
    <li>
        <b>Metric Design: </b>The design of the metrics should consider the following  
        <ol>
            <li>Model should leverage conversion rates for a set of attributes and use it to train the agent</li>
            <li>Model should factor the number of emails sent in addition to conversion rates to avoid outlier effect</li>
            <li>Metrics should track statistics such as Average, Min and Max conversion rates to understand the statistical distribution</li>
        </ol>
    </li>
    <li>
        <b>Metrics Evaluation: </b>The evaluation of the metrics need to factor the following
        <ol>
            <li>System should be able to retrain the model at prescribed intervals</li>
            <li>System should be able to replace the models used for training and predicting</li>
            <li>System should be able to monitor the performance of the model by running test transactions</li>
            <li>Actual conversion rates for a given set of attributes encoded in the state input should exceed the conversion goal</li>
        </ol>
    </li>
</ol>

### REQUIREMENTS

#### What are our (system) Assumptions?

Following are the assumptions made by the system

<ol>
    <li>All data needed for the system is collected and accessible by the system</li>
    <li>Changes in data such as additional attributes or value enumerations or data types are communicated in advance to the system in order to adapt and adjust before these changes are put into production (E.g. new email subjects)</li>
    <li>Sufficient validations are in place upstream to avoid incomplete or missing data</li>
    <li>The campaign data present is representative of the campaigns sent to the customers for the given subject ids</li>
</ol>


#### What are our (system) Requirements?

Based on the goals we come up with the following requirements 

<ol>
    <li>
        <b>Functional: </b>Here are the functional requirements of the system
        <ol>
            <li>System should advise a next action for a campaign manager to take as they decide what audience they should target</li>
            <li>System should enable the user (campaign manager) to see conversion rate likely for a given action</li>
            <li>System should allow authorized users (campaign managers) to test random states and action permutations to measure the performance of the system</li>
           <li>System should meet the desired conversion rate</li>
        </ol>
    </li>
    <li>
        <b>Non Functional Requirements: </b>Here are the non-functional requirements of the system
        <ol>
           <li>System should allow authorized users (machine learning engineers) to pre process data to be fed into a model for training to cover for situations when the logic of imputation has changed or upstream bugs are discovered</li>
            <li>System should allow authorized users (machine learning engineers) to change thresholds for the model and verify if the model meets the thresholds specified on validation data</li>
           <li>System should allow authorized users (machine learning engineers) to test different models</li>
           <li>System should allow authorized users (machine learning engineers) to retrain models on existing data</li>
        </ol>
    </li>
</ol>

### RISK & UNCERTAINTIES

#### What are the possible harms?

Here are some of the harms that we see 

<ol>
    <li>Poor Model Quality leads to actual conversion rates falling below desired</li>
    <li>Poor Model Quality leads to service provider's domain put on the spam list by major email service providers and / or by customers</li>
    <li>Customers have a bad perception of the service provider and it affects Brand</li>
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
        <b>Data Analysis:</b>In this phase sample datasets or subsets of sample data would be loaded into local development workspaces such as Jupyter notebooks to help understand the data. We will build visualizations and the purpose here would be to get some clues or intuition on what potentially could be affecting the end result. We have done that for this dataset in a notebook under <b>analysis/exploratory_data_analysis.ipynb</b>. In this notebook we are using direct pandas and exploring sample reinforcement learning
    </li>
    <li>
        <b>Dataset Processing:</b>We have leveraged a module <b>data_pipeline.py</b> with a class <b>ETL_Pipeline</b> which follows a Extract, Transform and Load cycle. During extract it reads training data from supplied <b>Email Campaign Data (sent_emails.csv, responded.csv, userbase.csv)</b>. In the <b>extract</b> phase (method) we load the raw data and create a dataframe. In the <b>transform</b> phase (method) we join the 3 files, perform transformations such as binning Tenure and Age, deriving Sent Day of the Week, etc and then drop columns we don't need. We encode all the values in numeric and save this as a transformed file namely email_campaign_data.csv as a part of <b>load</b> method
    </li>
    <li>
        <b>Data Partitioning:</b>We have leveraged a module <b>dataset.py</b> with a class <b>Email_Campaign_Dataset</b> which employs a Stratified K-Fold (default folds 5) to create training and testing data subsets. We use these subsets to provide the training and testing datasets to consumers of this class. There is also a get_validation_dataset method that applies a split from the training data as validation data. 
    </li>
    <li>
        <b>Model Training and Metrics</b>We have leveraged a module <b>metrics.py</b> with a class <b>Metrics</b> to return metrics on testing data as well as to generate a report. There is a notebook under <b>analysis/model_performance.ipynb</b> that leverages all these classes. Our primary metric is <b>Conversion Rate</b> for which we calculate the Mean, Median, Minimum and Maximum. The general idea is to leverage the Q Table to get the next action (and hence state) and then get the conversion rate at this new state
    </li>
    <li>
        <b>Deployment Strategy:</b>We have packaged the interface into a Flask app in the module <b>email_campaign_service.py</b>. This app exposes a REST API interface and performs the preprocessing and training of the model during the start up phase. It exposes 1 end points one for '/get-next-action' that takes in a payload for a state + action and returns the next action to take. We also have a <b>email_campaign_service_test_nb.ipynb</b> Notebook that starts this service to test it on local
    </li>
</ol>

#### High-level System Design

Our overall system design involves the following key components

<ol>
    <li>
        <b>Pipeline:</b>We will build a Machine Learning pipeline that has several steps and data can flow into and out from these steps where each step performs specific tasks such as pre-processing or training
    </li>
    <li>
        <b>Data Pre-processing:</b>This component will prepare the data so that it is usable for any learning process. This involves joining, some transforms and encoding. The final transformed data will flow through the Data Processing Pipeline to the next stage often saving the transformed files somewhere for efficient usages by the next stages in the pipeline
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
        <b>Customer Data Updates :</b>Updates to customer data for new customers and existing ones
    </li>
    <li>
        <b>New Ground Truth Data:</b>As new emails are sent their subject ids are updated along with the latest response data
    </li>
</ol>

#### Regulations

We have researched for following resources that speak to what a financial institution needs to be aware of and in compliance with 

<ol>
    <li>
        <b>CAN-SPAM:</b>https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business
    </li>
    <li>
        <b>Spam and malware:</b>https://crtc.gc.ca/eng/internet/anti.htm
    </li>
    <li>
        <b>UK:</b>https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/guide-to-pecr/electronic-and-telephone-marketing/electronic-mail-marketing/
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

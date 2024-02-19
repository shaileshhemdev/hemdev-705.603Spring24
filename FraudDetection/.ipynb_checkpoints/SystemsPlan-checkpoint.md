# Fraud Detection System Design

## MOTIVATION (Why's)

### Why are we solving this problem?

Almost all humans interact with Banks and in general Financial Institutions and they trust them with their money. Over the past several decades, the mode of paying for any goods and services has moved almost completely to electronic medium where humans are using debit cards, credit cards, money movement services such as Zelle, Paypal, etc to perform these transactions where Point of Sale (POS) systems hardly deal with cash. Internet has made humans purchase these goods and services at a rapid rate where the amount of financial transactions taking place on a daily basis is significantly more than what used to be generations ago. As such most human beings cannot always remember the purchases they made. Add to it, the electronic payment methods involve customers storing sensitive information including Personally Identifiable Information (PII) and Payment Card Industry (PCI) data in disparate systems all of which are vulnerable to security & data breaches. This has resulted in an increase in the probability of fraud where malicious actors obtain the PII and / or PCI information and make transactions that are not made by the actual customer and given that they put their trust in the financial institution, there is an implicit and often explicit expectation that the customers have from the financial insitutions to protect them from Fraud. 

#### Problem Statement

Every financial transaction goes through an authorization system that approves or declines the transaction. The key problem we are trying to solve is detecting fraudulent transactions at the time of authorizations and alerting the customers of potential fraud engaging them to let the transaction go through or not. 

#### Value Proposition

When the transaction is truly a fraud, the benefits to the customer are increased trust, protection from fraud and improved experience as they don't discover this after the fact requiring long customer support cycles. For the financial institution there are a long range of benefits such as reduced monetary risk, protection from reputation or brand damage and increased customer loyalty. 

### Why is our solution a viable one?

Identifying whether a fraudulent transaction during authorization is a viable one for the following reasons

<ol>
    <li>
        <b>Is Machine Learning the right fit?: </b> The variables that can determine whether a transaction is fraudulent or not are dynamic (constantly changing) and extensive making it hard to deterministically code in software. For example we cannot determine if a transaction is fraud purely on the basis of say the transaction amount or time of the day or merchant where it occured since the customer could be interacting with the specific merchants at those times for amounts in that range regularly. Malicious actors who commit this fraud are leveraging sensitive data which can often be purchased easily with breaches on the dark web to ensure that these transactions fit the patterns of the customer they are defrauding in order to go under the radar. Thus in order to determine whether a transaction is likely a fraud, we have to consider a wide variety of attributes and also similarities across millions of other customers to detect the subtelity in patterns that can predict if the transaction is likely fraud. To add to this these malicious actors are changing their tactics on a regular basis and so the system has to continuously adapt. Due to this it is not feasible to put deterministic business logic in the code that always arrives at the same result. What we need is a system that is good at determining these patterns and always learning to keep up with the pattern changes. 
    </li>
    <li>
        <b>Can we tolerate mistakes and a degree of risk?: </b> We first need to list the risks involved. The first risk is if we identify valid transactions as fraud. Most financial institutions do not approve such transactions and engage the customer. So the customer can provide the feedback if they made the transaction or not. If there are too many of these then it can cause nuisance to the customer even if they rationally appreciate the financial institution checking with them. The second risk is if a transaction if fraud but the financial insitution fails to detect this. This can cause a lot of grief to the customer including monetary loss and can lead to poor experience and reduced loyalty which in turn impacts the financial insitution. So there is a medium to high degree of risk involved if things are predicted incorrectly. However there is also an appetite to tolerate some mistakes. For example if occasionally a customer gets a request to confirm a valid transaction they are less likely to be irritated. On the other hand, they probably don't want to ever incur a fraudulent transaction. So while both factors are important, it is more important to get the fraudulent transactions right and then reduce the odds of flagging a valid transaction and this is the tolerance level
    </li>
    <li>
        <b>Is this solution (with ML) feasible? : </b> Machine Learning can do this as we have sophisticated algorithms for this pattern recognition or classification today coupled with the ability to analyze large amounts of data using cloud and distributed systems. Identifying whether a transaction is fraudulent or not is essentially a supervised learning problem specifically classification. In the absence of a machine learning system, customers would be calling in to indicate if the transaction is fraudulent and this data can be associated with the past transaction (annotated as fraud) and used for the learning process. Even going forward, since the system will engage the customer the customer's feedback can be used to perform this annotation to always have this data. As more data in terms of dimensions gets collected, the model can be enhanced. Cloud Storage has made it economically feasible to store large amounts of data in low cost commodity hardware storage as well as store structured and unstructured data. Technoligical advances also make it possible to have sophisticated ETL pipelines to collect raw data, cleanse it, perform transformatioms and store data with the right granularity, accuracy and completeness to enable a model to learn. 
    </li>
</ol>

## REQUIREMENTS (What's)

### SCOPE

#### What are our goals?

Following are the goals of the system

<ol>
    <li>
        <b>Organizational: </b> Identify a fraudulent transaction always and never mark a valid transaction as Fraud
    </li>
    <li>
        <b>System:</b>Efficient use of storage, low cost and great customer experience
    </li>
    <li>
        <b>User:</b>Users always alerted for fraudulent transaction before it occurs and never for valid transactions
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

#### High-level System Design

#### Development Workflow

### POLICY

#### Human-Machine Interfaces

#### Regulations

### OPERATIONS

#### Continuous Deployment

#### Post-deployment Monitoring

#### Maintenance

#### Quality Assurance

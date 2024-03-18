# Automated License Plate Registration (ALPR) System Design

## MOTIVATION (Why's)

### Why are we solving this problem?

Road transportation systems across the world involve multi lane and high speed toll roads where there is a charge to be paid by commuters who want speed and efficiency in their road travel. These toll charges enable appropriate maintenance of the toll highways. Having personnel man these toll exits to collect toll is both not feasible given the usage 24 x 7 and it also adds to the cost which in turn gets passed to the commuter in their toll charges. While many toll roads offer devices such as EZPass which when attached to the vehicle dashboard allows toll to be deducted as you pass the toll gates, many commuters might not have this such as commuters that are commuting out of state. 

In order to stay consisitent and efficient, toll check points need to employ systems that can read license plates as vehicles go through the gates, locate the vehicle registration, find the owner and then mail them the toll charges to pay offline. This also covers violations as well as situations where the placement of the devices like EZPass prevents their detection. 

#### Problem Statement

Detect the license plate number of the vehicle passing the toll checkpoints 

#### Value Proposition

With a good ALPR system, toll charges can be kept to a minimum, violations can be addressed and also long lines leading to the toll gates can be avoided

### Why is our solution a viable one?

We can evaluate if Detecting the License Plate Number is viable option using the following parameters

<ol>
    <li>
        <b>Is Machine Learning the right fit?: </b> Without machine learning, we would need personnel who can look at images taken and make notes of the license plate numbers. Or we would need humans manning the toll stations 24 x 7. The cost of having such personnel for a large number of toll booths will increase the cost of maintaining toll booths by a very high degree which in turn may increase the toll charges to a point where commuters would not want to use toll. Thus we need a way to solve this efficiently through automated means. Now without machine learning you cannot detect alphanumeric text from an image and thus machine learning is the right fit here
    </li>
    <li>
        <b>Can we tolerate mistakes and a degree of risk?: </b> In this application, the cost of mistake is high. A mistake is when the wrong license plate number is detected which can cause the wrong person to receive the payment charges. On the other hand if too many license plates are ignored then the toll company is set to lose money.  
    </li>
    <li>
        <b>Is this solution (with ML) feasible? : </b> In order to identify license plate number from the image we need algorithms that can mimic how the human brain processes images to find the object of interest. It knows where a license plate is and how it is different from its surrounding context. Within an image everything is pixel which can be encoded using colors and textures. Machine learning makes use of changes in colors, gradients and texttures to determine boundaries between objects. When you mark existing plates with bounding boxes, machine learning algorithms for visual search can identify what separates those boxes from surroundings since license plates are always at a certain location with a certain background and certains text / numbers inscribed on them. These algorithms use these patterns to find the area within the larger image that represents the license plate and then apply OCR algorithms to glean the text from the license plate image
    </li>
</ol>

## REQUIREMENTS (What's)

### SCOPE

#### What are our goals?

Following are the goals of the system

<ol>
    <li>
        <b>Organizational: </b> Identify a license plate always and never get an incorrect license plate determination
    </li>
    <li>
        <b>System:</b>Efficient processing of videos to obtain an optimal number of image frames needed
    </li>
    <li>
        <b>User:</b>Correct user is notified for toll charges based on the license plate number
    </li>
    <li>
        <b>Model:</b>Model should have high accuracy for license plate detection
    </li>
</ol>

#### What are the success criteria?

The above goals can be expressed in the following success criteria

<ol>
    <li>
        <b>Metric Design: </b>The design of the metrics should consider the following  
        <ol>
            <li>Model should be able to get a high degree of accuracy close to 100% in terms of license plate number resolution</li>
            <li>Model should be able to get the license plate object 90% of the time</li>
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
    <li>Cameras are able to capture the videos with needed resolution</li>
    <li>Sufficient guard rails are present upstream to foster trust in the data and ensuring data is accurate</li>
    <li>Video frame rates can be controlled to get the optimal number of images for the same license plate</li>
    <li>Network Bandwidth enables capturing the video streams and image streams at desired resolution</li>
    <li>A large amount of historical labelled (ground truth) data is available to initially train the models and this data is certified by the domain experts</li>
</ol>


#### What are our (system) Requirements?

Based on the goals we come up with the following requirements 

<ol>
    <li>
        <b>Functional: </b>Here are the functional requirements of the system
        <ol>
            <li>System should predict the license plate number</li>
            <li>System should be able to capture images from video frames</li>
            <li>System should detect license plate objects in images</li>
           <li>System should allow authorized users (machine learning engineers) to test different models</li>
           <li>System should allow authorized users (machine learning engineers) to retrain models on existing data</li>
           <li>System should have >90% Accuracy for License Plate Number Identification</li>
           <li>System should have >50% Precision for Object Detection</li>
        </ol>
    </li>
    <li>
        <b>Non Functional Requirements: </b>Here are the non-functional requirements of the system
        <ol>
           <li>System should allow authorized users (machine learning engineers) to pre process data to be fed into a model</li>
            <li>System should allow authorized users (machine learning engineers) to change thresholds for the model and verify if the model meets the thresholds specified on validation data</li>
        </ol>
    </li>
</ol>

### RISK & UNCERTAINTIES

#### What are the possible harms?

Here are some of the harms that we see 

<ol>
    <li>Poor Model Quality leads to wrong customers receiving toll charge notifications or duplicate charges</li>
    <li>Poor Model Quality leads to many license plates not detected leading to loss for Toll companies</li>
    <li>Certain types of vehicles or plates are vulnerable to inaccuracies</li>
    <li>Sudden dips and surges in the metrics impacting the overall performance of the model</li>
    <li>Model performance degrades over time</li>
</ol>


#### What are the causes of mistakes?

Here are some of the causes that lead to mistakes

<ol>
    <li>Data Systems are breached where the data used for training is corrupted or manipulated</li>
    <li>Bugs in upstream systems such as Cameras and Video Streaming systems lead to inaccurate or incomplete data</li>
    <li>Model performs well during training and even validation but not in actual field</li>
    <li>Model does not adapt to changing shape and quality of data</li>
</ol>

## IMPLEMENTATION (How's)

### DEVELOPMENT

#### Methodology

Our methodology involves the following Software Development Lifecycle processes 

<ol>
    <li>
        <b>Data Analysis:</b>In this phase sample datasets or subsets of sample data would be loaded into local development workspaces such as Jupyter notebooks to help understand the data. We will build visualizations and the purpose here would be to get some clues or intuition on what potentially could be affecting the end result. We have done that for this dataset in a notebook under <b>analysis/exploratory_data_analysis.ipynb</b>. In this notebook we are using test images to see the impact of cropping and other actions on license plate recognition
    </li>
    <li>
        <b>Dataset Processing:</b>We have leveraged a module <b>data_pipeline.py</b> with a class <b>Pipeline</b> which follows a Extract, Transform and Load cycle. During extract it reads image frames from a video stream such as <b>LicensePlateReaderSample_4K.mov</b> that is streamed by ffmpeg server. In the <b>extract</b> phase (method) save these images in a folder using <b>openccv</b>. In the <b>transform</b> phase (method) crop the images - given there are 2 lanes we create 2 cropped images one for the left lane and one for the right. We also only take part of the image since the top half shows vehicles that are far and that will appear closer in subsequent images. In the <b>load</b> phase (method) we read the cropped images and add it to a list
    </li>
    <li>
        <b>Data Partitioning:</b>We have leveraged a module <b>dataset.py</b> with a class <b>Object_Detection_Dataset</b> which employs a K-Fold (default folds 10) to create training and testing data subsets. The data is essentially all the cropped images and we do a split between these images storing the file paths as our features. <b>Note:</b> There is no predefined classes for the same
    </li>
    <li>
        <b>Model Training and Metrics</b>We have leveraged a module <b>metrics.py</b> with a class <b>Metrics</b> to return metrics on testing data as well as to generate a report. There is a notebook under <b>analysis/model_performance.ipynb</b> that leverages all these classes and simulates metrics. Of these <b>Precision</b> is critical for evaluating which model is good at object detection whereas <b>Accuracy</b> is good for license plate recognition. 
    </li>
    <li>
        <b>Model Selection Analysis:</b>In <b>analysis/model_performance.ipynb</b> we are focussing on Precision to evaluate the 2 Yolo models provided. We first run these on the 248 images we had and then also run it on the testing dataset using a K-fold
    </li>
    <li>
        <b>Deployment Strategy:</b>We have packaged the interface into a Flask app in the module <b>object_detection_service.py</b>. 
    </li>
</ol>

#### High-level System Design

Our overall system design involves the following key components

<ol>
    <li>
        <b>Pipeline:</b>We will build a Machine Learning pipeline that has several steps and data can flow into and out from these steps where each step performs specific tasks such as pre-processing or training
    </li>
    <li>
        <b>Data Pre-processing:</b>This component will prepare the data so that it is usable for any learning process. This involves cropping, resizing and splitting of the images. The final transformed images are saved and then loaded for subsequent steps
    </li>
    <li>
        <b>Object Detection:</b>We detect the bounding boxes and then crop the license plate images
    </li>
    <li>
        <b>Dataset:</b>This component provides the file paths for images to be considered for training, testing and validation
    </li>
    <li>
        <b>Training:</b>We are not training the model. The 2 supplied models are meant to be trained outside the scope of this system
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
        <b>High Resolution Cameras and Sensors:</b>We assume the cameras and sensor systems are able to provide a set of images related to the same license plate. This is important for the accuracy of the model since we want to arrive at our final prediction for license plate numbers using predictions from multiple images for the same license plate
    </li>
    <li>
        <b>Ground Truth Data:</b>We have some test images with bounding boxes labelled so that we can compare our model's bounding boxes with them.  
    </li>
</ol>

#### Regulations

We have researched for following resources that speak to what a financial institution needs to be aware of and in compliance with 

<ol>
    <li>
        <b>Automated License Plate Readers: State Statutes:</b>https://www.ncsl.org/technology-and-communication/automated-license-plate-readers-state-statutes
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
    <li>
        <b>FFMPEG Challenges:</b>The image frames captured from ffmpeg sometimes had blurring and distortion and it wasn't clear what was causing it. So this will need to be fixed
    </li>
    <li>
        <b>RTSP:</b>Enabling a RTSP interface where the system can listen to a real time video stream to produce the predictions
    </li>
</ol>


#### Post-deployment Monitoring

We will need to set up some monitors to get real time feedback on the following 

<ol>
    <li>
        <b>Data Quality:</b>Ensuring we get the same quality of images
    </li>
    <li>
        <b>Model Performance:</b>We try different OCR algorithms other than Tesseract
    </li>
</ol>

#### Maintenance

Classes are loosely coupled that allows us to change the ETL Pipeline from the Dataset Partitioning from the Model itself. This allows us to make changes to one of them without affecting the other. 

#### Quality Assurance

We have made use of notebooks to measure the quality 

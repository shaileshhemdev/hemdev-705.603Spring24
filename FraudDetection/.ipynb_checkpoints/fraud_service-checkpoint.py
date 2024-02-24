from flask import Flask
from flask import request
import os

from data_pipeline import ETL_Pipeline 
from dataset import Fraud_Dataset
from model import Fraud_Detector_Model
from metrics import Metrics

app = Flask(__name__)

# http://localhost:8786/infer?manufacturer=Audi&transmission=automatic&color=blue&engine=gasoline&drivetrain=all&state=owned&hasWarranty=True&odometer=12000&year=2020&bodytype=suv&price=20000

@app.route('/stats', methods=['GET'])
def getStats():
    return str(cf.model_stats())

@app.route('/infer', methods=['GET'])
def getInfer():
    args = request.args
    
    manufacturer = args.get('manufacturer')
    transmission = args.get('transmission')
    color = args.get('color')
    engine = args.get('engine')
    drivetrain = args.get('drivetrain')
    state = args.get('state')
    has_warranty = args.get('hasWarranty')
    bodytype = args.get('bodytype')
    odometer = int(args.get('odometer'))
    year = int(args.get('year'))
    price = int(args.get('price'))
    
    return cf.model_infer(manufacturer, transmission, color, engine, drivetrain, state, 
                          bodytype, has_warranty, odometer, year, price)

@app.route('/post', methods=['POST'])
def hellopost():
    args = request.args
    name = args.get('name')
    location = args.get('location')
    print("Name: ", name, " Location: ", location)
    imagefile = request.files.get('imagefile', '')
    print("Image: ", imagefile.filename)
    imagefile.save('/workspace/Hopkins/705.603Fall2023/workspace/ML_Microservice_Example/image.jpg')
    return 'File Received - Thank you'

if __name__ == "__main__":
    flaskPort = 8786
    
    # Process the Data needed to train the model
    dp = ETL_Pipeline('/workspace/shared-data/')
    df = dp.process('transactions-1.csv')

    # Initialize the metrics
    metrics = Metrics()

    # Create a fraud dataset with single fold
    fd = Fraud_Dataset(df,'is_fraud',1)

    # Obtain the training data
    X_train, y_train = fd.get_training_dataset(0)
    X_test, y_test = fd.get_testing_dataset(0)
    X_val, y_val = fd.get_validation_dataset(0)

    # Train the Model
    model = Fraud_Detector_Model()
    model.train(X_train, y_train, X_val, y_val)
    
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)


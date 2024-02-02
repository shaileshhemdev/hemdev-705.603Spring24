from flask import Flask
from flask import request
import os

from randomforest import randomforest

app = Flask(__name__)

#http://localhost:8786/infer?age=28&salary=40000

@app.route('/stats', methods=['GET'])
def getStats():
    return str(rf.model_stats())

@app.route('/infer', methods=['GET'])
def getInfer():
    args = request.args
    age = int(args.get('age'))
    salary = int(args.get('salary'))
    return rf.model_infer(age,salary)

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
    rf = randomforest()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)
    
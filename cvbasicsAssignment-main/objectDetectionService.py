from flask import Flask
from flask import request
import os

from carsfactors import carsfactors

app = Flask(__name__)

# Use postman to generate the post with a graphic of your choice

@app.route('/detect', methods=['POST'])
def detection():
    args = request.args
    name = args.get('name')
    location = args.get('description')
    
    imagefile = request.files.get('imagefile', '')
    print("Image: ", imagefile.filename)
    imagefile.save('LOCAL DIRECTORY')
    # The file is now downloaded and available to use with your detection class
    findings = ot.detect()
    # covert to useful string
    findingsString = 
    return findingsString

if __name__ == "__main__":
    flaskPort = 8786
    ot = ObjectDetection()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)


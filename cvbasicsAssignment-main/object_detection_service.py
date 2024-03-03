from flask import Flask
from flask import request, jsonify
import sys
import os

from object_detection import ObjectDetection

app = Flask(__name__)

# Use postman to generate the post with a graphic of your choice

@app.route('/detect', methods=['POST'])
def detection():
    imagefile = request.files.get('imagefile', '')
    print("Image: ", imagefile.filename)
    ot.save_source_image(imagefile.filename, imagefile)

    # The file is now downloaded and available to use with your detection class
    return ot.detect(imagefile.filename)

if __name__ == "__main__":
    flaskPort = 8786

    # Get command line arguments
    if (len(sys.argv)>1):
        data_folder                 = sys.argv[1]
    else: 
        data_folder = os.environ['data-folder']

    ot = ObjectDetection(base_dir=data_folder)
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)


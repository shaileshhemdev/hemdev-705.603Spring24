from metrics import Metrics
import cv2 as cv
import numpy as np
import pytesseract
import re
from data_pipeline import Pipeline
from metrics import Metrics
import sys
import os

class Object_Detection_Model:
    """
    A class used to represent the Object Detection Model 

    ...

    Attributes
    ----------
    _cfg_file : str
        Model Configuration File (Yolo Config)
    _weights_file : str
        Weights for the Yolo model
    base_dir : str
        Base directory for the data, output and model files
    confidence_threshold : str
        Threshold for confidence for bounding boxes. Defaulted to 0.5
    box_threshold : float
        Threshold for separation of bounding boxes. Defaulted to 0.4
    scaling_factor : float
        Color scaling factor
    _model_dir : str
        Folder for the model files
    _images_dir : str
        Full path for the image files
    width : int
        Width of the images used for the model
    height : int
        Height of the images used for the model
    net : cv.dnn.readNet
        Deep neural netowrk used to train the model
    output_layer : cv.dnn.readNet
        Output layer

    Methods
    -------
    test()
        Predict class for test data and return metrics 

    """
    def __init__(self, cfg_file, weights_file, base_dir = '', model_folder='models/', images_folder = 'cropped/', confidence_threshold=0.5, box_threshold = 0.4, scaling_factor = 1/255):
        """ Initializes the Object Detection Model

        Parameters
        ----------
        cfg_file : str
            Name of the Yolo 5 Config file to use
        weights_file : str
            Name of the Yolo 5 Weights file to use
        base_dir : str
            Directory from where the images and the model files can be accessed
        confidence_threshold : float
            Confidence threshold for object detections     
        box_threshold : float
            Box threshold for object detections   
        scaling_factor : float
            Scaling factor

        """
        # Initialize instance variables
        self._cfg_file = cfg_file
        self._weights_file = weights_file
        self.base_dir = base_dir
        self.confidence_threshold = confidence_threshold
        self.box_threshold = box_threshold
        self.scaling_factor = scaling_factor
        self._model_dir = base_dir + model_folder
        self._images_dir = base_dir + images_folder

        # We have chosen these values to work with yolo
        self.width = 608
        self.height = 608
        
        # Load Yolo
        self.net = cv.dnn.readNet(self._model_dir + self._weights_file, self._model_dir + self._cfg_file)
        
        # Initialize Neural Network
        layer_name = self.net.getLayerNames()
        self.output_layer = [layer_name[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_licence_plate(self, img_cv):   
        """ Get the license plate number
        Parameters
        ----------
        img_cv : bytes
            Actual image of the license plate

        Return
        ----------
        license_plate_no : str
            Predicted license plate number
        """
        # NOT USING THIS Resize the image slighty to see if it covers slightly misoriented values
        img_resized = cv.resize(img_cv, None, fx = 2, fy = 2,  interpolation = cv.INTER_CUBIC)

        # Convert to grayscale
        img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
        
        # Apply dilation to reduce noise
        kernel = np.ones((1, 1), np.uint8)
        img_gray = cv.dilate(img_gray, kernel, iterations=1)
        img_gray = cv.erode(img_gray, kernel, iterations=1)
        
        # Blur and Filter
        cv.threshold(cv.medianBlur(img_gray, 3), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        cv.adaptiveThreshold(cv.bilateralFilter(img_gray, 9, 75, 75), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 2)

        # Predict using OCR
        prediction = pytesseract.image_to_string(img_gray, lang ='eng', config ='--oem 3 --psm 6 ')
        
        # Remove everything outside of alphanumeric characters
        license_plate_no = re.sub(r'[^A-Z0-9]+', '', prediction)
        
        # License plates are also meant to be between 6 and 8 characters
        if (len(license_plate_no)<6 or len(license_plate_no)>8):
            return ''
        
        return license_plate_no 

    def test(self, license_plate_no, image_name_arr):
        """ Run a series of images through to determine the best prediction
        Parameters
        ----------
        license_plate_no : str
            Actual license plate no
        image_name_arr : ndarray
            Array of file names

        Return
        ----------
        lp_best, score : tuple
            Tuple of license plate predicted along with its Levenshtein distance from the actual

        """
        # Initialize the metrics 
        metrics = Metrics()

        # Initialize list of predictions
        lp_predictions = list()

        # Loop through the list of images
        for img_name in image_name_arr:
            # Load the image
            img = cv.imread(self._images_dir + img_name)
            img = cv.resize(img, (608, 608))

            # Get the bounding box confidences
            box = self.get_box_confidences(img)

            # If atleast 1 bounding box is found 
            if (len(box))>=1:
                # Get the first box
                dim = box[0][0]
                
                # Get the actual license plate image
                lp_img = img[dim[1]:dim[1]+dim[3],dim[0]:dim[0]+dim[2]]

                # Run OCR Detection
                lp = self.detect_licence_plate(lp_img)
                
                # Add to the list if its not ''
                if (len(lp)>=1):
                    lp_predictions.append(lp)
        
        # Get the plate with the highest frequency
        lp_unique, lp_frequency = np.unique(np.array(lp_predictions), return_counts = True)
        lp_best = ''     
        if (len(lp_unique)> 0):
            lp_best = lp_unique[np.argmax(lp_frequency)]

        # Calculate Levinshtein distance between actual and predicted license plate numbers
        return (lp_best, metrics.get_distance(license_plate_no, lp_best),lp_predictions)

    def get_box_confidences(self, img):
        """ Run a single image against the model

        Parameters
        ----------
        image_name : str
            Name of the file
        license_plate_classify : ndarray
            Array for each object if its a license plate or not

        """
        # Provide image to model 
        blob = cv.dnn.blobFromImage(img, self.scaling_factor , (self.width, self.height), True, False)  

        # Detect objects
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layer)

        # Initialize arrays for class ids (object class ids), confidence for each and box values to print in the image
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    # Object detection
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] *self.height)

                    # Reactangle Coordinate
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    # Populate boxes
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

         # Remove overlapping boxes
        indexes = cv.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.box_threshold)

        # Create dictionary of class label and associated confidence
        box_detection_confidence = []

        # Run through the range and populate the dictionary
        for i in range(len(boxes)):
            if i in indexes:
                box_detection_confidence.append(tuple((boxes[i], confidences[i]*100)))
        
        return box_detection_confidence 

    def predict(self, resized_img_list, prediction_threshold):
        """ Predict the license nos

        Parameters
        ----------
        resized_img_list : ndarray
            List of resized images
        prediction_threshold : str
            Threshold to consider to pick the potential license plates
        """
        # Initialize list of predictions
        lp_predictions = list()
         # Loop through the list of images
        for img in resized_img_list:
            # Get the bounding box confidences
            box = self.get_box_confidences(img)

            # If atleast 1 bounding box is found 
            if (len(box))>=1:
                # Get the first box
                dim = box[0][0]
                
                # Get the actual license plate image
                lp_img = img[dim[1]:dim[1]+dim[3],dim[0]:dim[0]+dim[2]]

                # Run OCR Detection
                lp = self.detect_licence_plate(lp_img)
                
                # Add to the list if its not ''
                if (len(lp)>=1):
                    lp_predictions.append(lp)
        
        # Get the plate with the highest frequency
        lp_unique, lp_frequency = np.unique(np.array(lp_predictions), return_counts = True)
        detected_license_plates = lp_unique[np.argwhere(lp_frequency > prediction_threshold)]

        # Return the predicted license plates
        return detected_license_plates


# Example usage
if __name__ == "__main__":
    # Get command line arguments
    img_counter = 1
    if (len(sys.argv)>1):
        video_stream_url  = sys.argv[1]
        data_folder  = sys.argv[2]
        predictions_run_folder = sys.argv[3]
        original_image_folder = sys.argv[4]
        cropped_image_folder = sys.argv[5]
        models_folder = sys.argv[6]
        prediction_threshold = sys.argv[7]
        img_counter  = sys.argv[8]
        dataset_name  = "License Plate Detection"
    else: 
        video_stream_url = os.environ['video-stream-url']
        data_folder  = os.environ['video-stream-base-folder']
        predictions_run_folder = os.environ['predictions-run-folder']
        original_image_folder = os.environ['original-image-folder']
        cropped_image_folder = os.environ['cropped-image-folder']
        models_folder = os.environ['models-folder']
        prediction_threshold = os.environ['prediction_threshold']
        dataset_name  = "License Plate Detection"
    
    print("Starting process")
    #in_file = 'udp://127.0.0.1:23000'  # Example UDP input URL
    width = 3840  # Example width
    height = 2160  # Example height

    # First extract, transform and load
    pipeline = Pipeline(data_folder,video_stream_url,predictions_run_folder,original_image_folder,cropped_image_folder,models_folder)
    pipeline.extract(int(img_counter))
    pipeline.transform()
    resized_img_list = pipeline.load()

    # Initialize the model
    model_base_dir = data_folder  
    model = Object_Detection_Model(cfg_file= "lpr-yolov3.cfg", weights_file='lpr-yolov3.weights', 
                                base_dir=model_base_dir, model_folder=models_folder, images_folder = cropped_image_folder)
    
    predicted_license_nos = model.predict(resized_img_list,int(prediction_threshold))
    print(predicted_license_nos)


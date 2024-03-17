from data_pipeline import Pipeline 
from metrics import Metrics
import cv2 as cv
import numpy as np

class Object_Detection_Model:
    """
    A class used to represent the Object Detection Model 

    ...

    Attributes
    ----------
    cls : RandomForestClassifier
        Classifier used for making prediction on whether a transaction is fraudulent or not

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
        self.width = 416
        self.height = 416
        
        # Load Yolo
        self.net = cv.dnn.readNet(self._model_dir + self._weights_file, self._model_dir + self._cfg_file)
        
        # Initialize Neural Network
        layer_name = self.net.getLayerNames()
        self.output_layer = [layer_name[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def test(self, image_name, license_plate_classify):
        """ Test the Model 

        Parameters
        ----------
        image_name : str
            Name of the file
        license_plate_classify : ndarray
            Array for each object if its a license plate or not

        """
         # Load the image
        img = cv.imread(self._images_dir + image_name)
        img = cv.resize(img, (416, 416))

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


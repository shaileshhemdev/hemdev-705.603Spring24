import cv2 as cv
import numpy as np

class ObjectDetection:
    def __init__(self, base_dir = '', confidence_threshold=0.5, box_threshold = 0.4, scaling_factor = 1/255):
        """
        A class used to represent the Core Object Detection Processing

        ...

        Attributes
        ----------
        base_dir : str
            The directory where the files such as weights, configurations and images need to be stored
        confidence_threshold : float
            The threshold for confidence for an object detection defaulting to 0.5
        box_threshold : float
            The threshold for boxes for an object detection defaulting to 0.4 while removing overlapping boxes
        scaling_factor : float
            The scaling factor - defaulting to 1/255 since we have RGB color code values

        Methods
        -------
        process()
            Key method that takes the image saved, obtains the various classes, their confidence intervals and boxes for objects
        get_objects()
            Given the classes, boxes it removes overlapping redundant boxes. As a result you get the exact number of objects found
        save_source_image()
            Saves source image in a directory
        """
    
        # Initialize instance variables
        self.base_dir = base_dir
        self.confidence_threshold = confidence_threshold
        self.box_threshold = box_threshold
        self.scaling_factor = scaling_factor

        # We have chosen these values to work with yolo
        self.width = 416
        self.height = 416

        # Load Yolo

        self.net = cv.dnn.readNet(self.base_dir + 'yolov3.weights', self.base_dir + "yolov3.cfg")

        # Get classes and layers
        self.classes = []
        with open("coco.names", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Initialize Neural Network
        layer_name = self.net.getLayerNames()
        self.output_layer = [layer_name[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def save_source_image(self, image_name, image_byte_stream):
        """
        A method used to save source image

        ...

        Parameters
        ----------
        image_name : str
            The name of the image
        image_byte_stream : bytes
            The bytes for the image to be saved
        """
        image_byte_stream.save(self.base_dir + image_name)

    def detect(self, image_name):
        """
        A method used to take an image and detect objects
        ...

        Parameters
        ----------
        class_ids : ndarray
            Array of class labels found through the object detection
        boxes : ndarray
            Array of boxes found through the object detection
        confidences : ndarray
            Array of confidences found through the object detection   

        Returns
        ----------
        class_confidence : ndarray
            Array of tuples with the first element being the class label and the second being the confidence %                     
        """
        class_ids, boxes, confidences = self.process(image_name)
        return self.get_objects(class_ids, boxes, confidences)

    def process(self, image_name):
        """
        A method used to process the image through the Neural Network for Object detection and return the classes, boxes and 
        confidence for those class predictions 
        ...

        Parameters
        ----------
        image_name : str
            The name of the image
        
        Returns
        ----------
        class_ids : ndarray
            Array of class labels found through the object detection
        boxes : ndarray
            Array of boxes found through the object detection
        confidences : ndarray
            Array of confidences found through the object detection                        
        """
        # Load the image
        img = cv.imread(self.base_dir + image_name)

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
           
        return (class_ids, boxes, confidences)
    
    def get_objects(self, class_ids, boxes, confidences): 
        """
        A method used to remove redundant or overlapping boxes. As a result you get the exact number of objects found with the confidence
        ...
        Parameters
        ----------
        class_ids : ndarray
            Array of class labels found through the object detection
        boxes : ndarray
            Array of boxes found through the object detection
        confidences : ndarray
            Array of confidences found through the object detection  
        
        Returns
        ----------
        class_confidence : ndarray
            Array of tuples with the first element being the class label and the second being the confidence %  
        """
        # Remove overlapping boxes
        indexes = cv.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.box_threshold)

        # Create dictionary of class label and associated confidence
        class_confidence = []

        # Run through the range and populate the dictionary
        for i in range(len(boxes)):
            if i in indexes:
                label = str(self.classes[class_ids[i]])
                #class_confidence += {label, confidences[i]*100}
                class_confidence.append(tuple((label, confidences[i]*100)))
        
        return class_confidence


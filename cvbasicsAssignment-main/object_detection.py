import cv2 as cv
import numpy as np

class ObjectDetection:
    def __init__(self, base_dir = '', confidence_threshold=0.5, box_threshold = 0.4, scaling_factor = 1/255):
        print('Init')

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
        image_byte_stream.save(self.base_dir + image_name)

    def process(self, image_name):
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
        # Remove overlapping boxes
        indexes = cv.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.box_threshold)

        # Create dictionary of class label and associated confidence
        class_confidence = []

        # Run through the range and populate the dictionary
        for i in range(len(boxes)):
            if i in indexes:
                label = str(self.classes[class_ids[i]])
                class_confidence += (label, confidences[i]*100)
        
        return class_confidence


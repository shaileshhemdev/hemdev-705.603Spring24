import numpy as np
import cv2 as cv
from pylabel import importer
import sys
import os
import pytesseract
import re
import deployment_udp_client

class Pipeline:
    """
    A class used to represent the Data Pipeline

    ...

    Attributes
    ----------
    _data_folder : str
        The name of the main folder where the content is. We expect the images and annotations to be stored separately in different sub folders
    _dataset_name : str
        The name of the dataset
    _images_folder : str
        The name of the images sub folder
    _annotations_folder : str
        The name of the annotations sub folder
    _yolo_sub_folder : str
        The name of the yolo sub folder
    _coco_sub_folder : str
        The name of the yolo sub folder

    Methods
    -------
    convert_pascal_to_yolo_format()
       Converts a given file to Yolo format
    convert_pascal_to_coco_format()
       Converts a given file to Coco format
    get_licence_plate_image():
        Crops the license plate image
    detect_licence_plate():
        Given a cropped image detects the license plate no

    """
    
    def __init__(self, data_folder, video_stream_url, output_images_folder="alpr-images", original_image_folder="/original", cropped_image_folder="/cropped", 
                 dataset_name="License Plate Recognition", images_sub_folder="images", annotations_sub_folder="annotations", 
                 yolo_sub_folder="yolo/labels", coco_sub_folder="coco", image_width = 3840, image_height = 2160):
        """ Initializes the Data Pipeline Class

        Parameters
        ----------
        data_folder : str
            The name of the main folder where the content is. We expect the images and annotations to be stored separately in different sub folders
        dataset_name : str
            The name of the dataset
        images_sub_folder : str
            The name of the images sub folder
        annotations_sub_folder : str
            The name of the annotations sub folder
        yolo_sub_folder : str
            The name of the yolo sub folder
        coco_sub_folder : str
            The name of the yolo sub folder
        """
        self._data_folder = data_folder
        self._images_folder =  images_sub_folder
        self._annotations_folder = self._data_folder + annotations_sub_folder
        self._coco_sub_folder = self._data_folder + coco_sub_folder
        self._yolo_sub_folder = self._data_folder + yolo_sub_folder
        self._dataset_name = dataset_name
        self._video_stream_url = video_stream_url
        self._license_plate_images = data_folder + output_images_folder 
        self._original_image_folder = original_image_folder 
        self._cropped_image_folder = cropped_image_folder
        self._image_width = image_width
        self._image_height = image_height

    def extract(self, img_counter=1):
        """ Reads the video stream to load the images

        """
        deployment_udp_client.stream_video(self._video_stream_url,self._license_plate_images + self._original_image_folder, self._image_width, self._image_height,img_counter)

    def transform(self):
        """ Load all the raw images

        """
        # Get the directory for raw images
        directory =  self._license_plate_images + self._original_image_folder

        # Get directory for cropped images
        cropped_image_dir = self._license_plate_images + self._cropped_image_folder
        
        # Run a loop of all images
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
    
            # checking if it is a file
            if os.path.isfile(f):
                original_img = cv.imread(f)
                
                if original_img is not None:
                    # Crop the image to what we have seen as good dimension
                    #print(filename)
                    cropped_image_left = original_img[1500:3000 ,600 :1300 ]
                    cropped_image_right = original_img[1500:3000 ,1300 :3000 ]
                    
                    # Image name split
                    img_name_arr = filename.split('.')
                    img_name = img_name_arr[0]
                    img_ext = img_name_arr[1]

                    # Cropped image path
                    cropped_img_left_file_name = cropped_image_dir + "/" + img_name + "_left." + img_ext
                    cropped_img_right_file_name = cropped_image_dir + "/" + img_name + "_right." + img_ext

                    # Save the cropped image
                    cv.imwrite(cropped_img_left_file_name, cropped_image_left)
                    cv.imwrite(cropped_img_right_file_name, cropped_image_right)


    def load(self):
        """ Load all the raw images

        """
        resized_images = list()

        # Get the directory for raw images
        directory =  self._license_plate_images + self._cropped_image_folder

        # Run a loop of all images
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
    
            # checking if it is a file
            if os.path.isfile(f):
                cropped_img = cv.imread(f)
                
                if cropped_img is not None:
                    resized_image = cv.resize(cropped_img, (608, 608)) 
                    resized_images.append(resized_image)
        
        return resized_images

    def convert_pascal_to_yolo_format(self, source_file):
        """ Executes the Pipeline to return yolo text file
        
        Parameters
        -------
        source_file : str
            The name of the source file

        """
        # Check if annotation is found 
        is_annotation_found = True

        # Load the dataset
        dataset = importer.ImportVOC(path=self._annotations_folder, path_to_images=self._images_folder, name=self._dataset_name)

        # Extract the file name w/o the file extension and use it to form the name of the json file
        if (source_file is None):
            output_file_name = self._yolo_sub_folder
        else:
            file_name_array = source_file.split(".")
            output_file_name = file_name_array[0] + ".txt"
            annotation_file_name = file_name_array[0] + ".xml"

            # Check if we have annotations file
            f = os.path.join(self._annotations_folder, annotation_file_name)
            if os.path.isfile(f):
                try:
                    dataset.df = dataset.df[dataset.df.img_filename.isin([source_file])].reset_index()
                except:
                    is_annotation_found = False
            else:   
                is_annotation_found = False
        
        # Export the text file
        if is_annotation_found:
            dataset.export.ExportToYoloV5(output_path=self._yolo_sub_folder)[0]  

    def convert_pascal_to_coco_format(self, source_file=None):
        """ Executes the Pipeline to return transformed dataset 
        
        Parameters
        -------
        source_file : str
            The name of the source file

        """
        # Check if annotation is found 
        is_annotation_found = True

        # Load the dataset
        dataset = importer.ImportVOC(path=self._annotations_folder, path_to_images=self._images_folder, name=self._dataset_name)

        # Extract the file name w/o the file extension and use it to form the name of the json file
        if (source_file is None):
            output_file_name = self._dataset_name + ".json"
        else:
            file_name_array = source_file.split(".")
            output_file_name = file_name_array[0] + ".json"
            annotation_file_name = file_name_array[0] + ".xml"

            # Check if we have annotations file
            f = os.path.join(self._annotations_folder, annotation_file_name)
            if os.path.isfile(f):
                try:
                    dataset.df = dataset.df[dataset.df.img_filename.isin([source_file])].reset_index()
                except:
                    is_annotation_found = False
            else:   
                is_annotation_found = False
        
        if is_annotation_found:
            dataset.export.ExportToCoco(output_path=self._coco_sub_folder + "/" + output_file_name)[0]   

    def get_licence_plate_image(self, source_file=None, images_folder = "images/", cropped_path="cropped"):    
        if (source_file is not None):
            # Get the annotations file
            file_name_array = source_file.split(".")
            img_name = file_name_array[0]
            img_extension = file_name_array[1]
            annotation_file_name = img_name + ".json"
            annotations_path = self._coco_sub_folder + "/" + annotation_file_name

            # We will expect the coco files to be present
            dataset_coco = importer.ImportCoco(annotations_path, path_to_images=self._images_folder, name=self._dataset_name)

            img_row = dataset_coco.df.loc[dataset_coco.df['img_filename'] == source_file]
            x = int(img_row['ann_bbox_xmin'])
            y = int(img_row['ann_bbox_ymin'])
            w = int(img_row['ann_bbox_xmax'])
            h = int(img_row['ann_bbox_ymax'])

            # Path to image
            img_path = self._data_folder + images_folder + source_file
            img = cv.imread(img_path)

            # Write cropped image
            cropped_image = img[y:h, x :w]
            cropped_img_file_name = self._data_folder + cropped_path + "/" + img_name + "_cropped." +  img_extension 
            cv.imwrite(cropped_img_file_name, cropped_image)
            return cropped_img_file_name

    def detect_licence_plate(self, source_file=None, images_folder = "cropped/"):    
        if (source_file is not None):
            # Read the image
            img_path = self._data_folder + images_folder + source_file
            img_cv = cv.imread(img_path)

            # NOT USING THIS Resize the image slighty to see if it covers slightly misoriented values
            img_resized = cv.resize(img_cv, None, fx = 2, fy = 2,  interpolation = cv.INTER_CUBIC)

            # Convert to grayscale
            img_gray = cv.cvtColor(img_cv, cv.COLOR_BGR2GRAY)

            # Predict using OCR
            prediction = pytesseract.image_to_string(img_gray, lang ='eng', config ='--oem 3 --psm 8 ')

            # Extract the text between first 
            license_plate_no = prediction
            
            # Get the first alpha numeric character
            start = self.letter_or_digit(license_plate_no)
            license_plate_no = license_plate_no[start:]

            # Remove everything outside of alphanumeric characters letters for license plate are upper case
            license_plate_no = re.sub(r'[^A-Z0-9-]+', '', license_plate_no)
            return license_plate_no 

# Example usage
if __name__ == "__main__":
    # Get command line arguments
    if (len(sys.argv)>1):
        video_stream_url  = sys.argv[1]
        data_folder  = sys.argv[2]
        img_counter  = sys.argv[3]
        dataset_name  = "License Plate Detection"
    else: 
        video_stream_url = os.environ['video-stream-url']
        data_folder  = os.environ['video-stream-image-folder']
        dataset_name  = "License Plate Detection"
    
    print("Starting process")
    #in_file = 'udp://127.0.0.1:23000'  # Example UDP input URL
    width = 3840  # Example width
    height = 2160  # Example height

    pipeline = Pipeline(data_folder,dataset_name,video_stream_url)
    pipeline.extract(int(img_counter))
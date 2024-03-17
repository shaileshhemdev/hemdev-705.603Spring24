from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import os
import numpy as np

class Object_Detection_Dataset:
    """
    A class used to represent the Object Detection Dataset 

    ...

    Attributes
    ----------
    X : object
        an array holding the feature values
    y : object
        an array holding the indexes
    _image_folder : str
        The folder where the images are
    _training_sets : dict
        the dictionary holding training data sets by fold as key
    _testing_sets : dict
        the dictionary holding testing data sets by fold as key

    Methods
    -------
    get_training_dataset()
        Get Training data for the specified fold 
    get_testing_dataset()
        Get Testing data for the specified fold 
    get_validation_dataset()
        Get Validation data for the specified fold 
    """
    
    def __init__(self, data_folder, n_folds=5):
        """ Initializes the Fraud_Dataset Class

        Parameters
        ----------
        data_folder : str
            The folder where all the images are stored which is our core dataset
        n_folds : int
            The number of folds needed from the data

        """
        self._image_folder = data_folder 

        # Load the list of file names 
        image_file_names = list()
        for filename in os.listdir(self._image_folder):
            image_file_names.append(filename)

        self.X = np.array(image_file_names)
        self.y = range(len(image_file_names))

        # Perform a K Fold 
        kf = KFold(n_splits=n_folds, random_state=None, shuffle=False)

        # Initialize the training and testing sets
        self._training_sets = {}
        self._testing_sets = {}

        # Perform K Fold and build the dictionary
        for i, (train_index, test_index) in enumerate(kf.split(self.X, self.y)):
            #print(f"Fold {i}:")
            #print(f"  Train: index={train_index}")
            training_set = self.X[train_index]
            testing_set = self.X[test_index]
            self._training_sets[i] = training_set
            self._testing_sets[i] = testing_set

        #print(self._training_sets)
    
    def get_training_dataset(self, fold):
        """ Get the Training Dataset

        Parameters
        ----------
        fold : int
            The fold for which training data is desired

        """
        return self._training_sets[fold]

    def get_testing_dataset(self, fold):
        """ Get the Testing Dataset

        Parameters
        ----------
        fold : int
            The fold for which testing data is desired

        """
        return self._testing_sets[fold]
    
    def get_validation_dataset(self, fold, val_split=0.3):
        """ Get the Validation Dataset

        Parameters
        ----------
        fold : int
            The fold for which validation data is desired

        """
        x_train, y_train = self._training_sets[fold]
        x_train, x_vals, y_train,y_val  = train_test_split(x_train, y_train , 
                                                                           test_size = val_split, 
                                                                           stratify=y_train, random_state=0)
        return x_vals, y_val
       
    
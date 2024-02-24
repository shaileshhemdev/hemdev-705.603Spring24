from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

class Fraud_Dataset:
    """
    A class used to represent the Fraud Dataset 

    ...

    Attributes
    ----------
    X : object
        an array holding the feature values
    y : object
        an array holding the class label values
    _df : df
        the dataframe representing the fraud dataset
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
    
    def __init__(self, transformed_df, class_label_col, n_folds=5):
        """ Initializes the Fraud_Dataset Class

        Parameters
        ----------
        transformed_df : df
            The dataset (Pandas Dataframe) that represents the final cleansed and preprocessed data
        class_label_col : str
            The name of the column that holds the class label 
        n_folds : int
            The number of folds needed from the data

        """
        self._df = transformed_df
        self.X = self._df.loc[:, self._df.columns != class_label_col].values
        self.y = self._df[class_label_col].values.ravel()

        # Perform a K Fold 
        skf = StratifiedKFold(n_splits=n_folds)

        # Initialize the training and testing sets
        self._training_sets = {}
        self._testing_sets = {}

        # Perform K Fold and build the dictionary
        for i, (train_index, test_index) in enumerate(skf.split(self.X, self.y)):
            #print(f"Fold {i}:")
            #print(f"  Train: index={train_index}")
            training_tuple = (self.X[train_index], self.y[train_index])
            testing_tuple = (self.X[test_index], self.y[test_index])
            self._training_sets[i] = training_tuple
            self._testing_sets[i] = testing_tuple

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
       
    
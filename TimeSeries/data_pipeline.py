import numpy as np
import pandas as pd
from datetime import datetime, date 
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

class ETL_Pipeline:
    """
    A class used to represent the Data Pipeline

    ...

    Attributes
    ----------
    _data_folder : str
        a string used to store the folder for the input and intermediate transformed data
    _source : str
        the name of the source file
    source_df : df
        the dataframe representing the source data
    transformed_df : df
        the dataframe representing the transformed data

    Methods
    -------
    process()
        Key method that extracts, transforms and loads the source. Optimizes by ensuring that if transformed
        data is present then it simply reads it back instead of reading source, applying transforms and writing
        transformed file
    extract()
        Reads the source file given the directory and file name. Expects the file to be a CSV
    transform()
        Performs Transformations on the source file to produce final features. As a part of transformation
        it creates new derived attributes (as advised from data analysis), removes unncessary columns, 
        performing scaling and encoding operations
    load()
        Saves the final transformed file
    """
    
    def __init__(self, data_folder):
        """ Initializes the Data Pipeline Class

        Parameters
        ----------
        source_file : str
            The name of the source file
        data_folder : str
            The folder with the source file is kept

        """
        self._data_folder = data_folder
        self.source_df = None
        self.transformed_df = None
    
    def process(self, source_file):
        """ Executes the Pipeline to return transformed dataset 
        
        Returns
        -------
        transformed_df
            The dataset (Pandas Dataframe) equivalent of the transformed file

        """
        try:
            transformed_df = pd.read_csv(self._data_folder + "forecasting_history.csv")
        except Exception:
            print("Did not find the forecasting_history.csv")
            source_df = self.extract(source_file)
            transformed_df = self.transform(source_df)
            self.load()   
        
        return transformed_df

    def extract(self, source_file):
        """ Reads the source to return source dataset 
        
        Returns
        -------
        source_df
            The dataset (Pandas Dataframe) equivalent of the source file

        """
        self.source_df = pd.read_csv(self._data_folder + source_file)
        return self.source_df

    def transform(self, source_df):
        """ Transforms the source to return transformed dataset 
        
        Returns
        -------
        transformed_df
            The dataset (Pandas Dataframe) after performing transformations on the source data

        """
        # Let's convert transaction date and time and dob to date-time
        source_df["dob_dt"] = pd.to_datetime(source_df['dob'], format='%Y-%m-%d', errors='coerce')
        source_df["txn_dt"] = pd.to_datetime(source_df['trans_date'], format='%Y-%m-%d', errors='coerce')

        # Compute weekday from transaction date
        source_df['txn_weekday'] = source_df['txn_dt'].dt.day_name()

        # Compute age from date of birth 
        source_df['age'] = source_df['dob'].apply(ETL_Pipeline.age) 

        # Compute hour from transaction date
        source_df['txn_hour'] = source_df['txn_dt'].dt.hour

        # Slot the times into well known time ranges
        time_ranges = [0,4,8,12,16,21,24]
        part_of_day_dict = ['Late Night', 'Early Morning','Morning','Afternoon','Evening','Night']
        source_df['part_of_day'] = pd.cut(source_df['txn_hour'], bins=time_ranges, labels=part_of_day_dict, include_lowest=True)

        # Compute month from transaction date
        source_df['txn_month'] = source_df['txn_dt'].dt.month_name()

        # Compute distance between customer's location and merchant location 
        source_df['distance_from_merchant'] = self.haversine_vectorize(source_df['lat'],source_df['long'],
                                                                       source_df['merch_lat'],source_df['merch_long'])
        source_df['txn_dist'] = source_df["distance_from_merchant"].apply(ETL_Pipeline.classify_distance) 

        # Drop the above columns
        cols_to_drop = ['ssn','acct_num','cc_num','person','first','last','dob','gender','street','city','state','zip','person_loc','age',
                'merchant_loc','lat','long','trans_num','trans_date', 'trans_time','unix_time','txn_time','distance_from_merchant','txn_month',
                'dob_dt','txn_hour','address','category','merch_lat','merch_long','normalized_job', 'merchant','job','city_pop']

        trimmed_df = source_df.drop(columns=cols_to_drop,errors='ignore')

        # Drop the column with the serial number
        for col in trimmed_df.columns:
            idx = source_df.columns.get_loc(col)
            if idx == 0:
                trimmed_df = trimmed_df.drop(columns=source_df.columns[idx])
        
        # Rename column
        base_df = trimmed_df.rename(columns={'txn_dt':'ds'})

        # Generate Final features 
        self.transformed_df = base_df
        return self.transformed_df
    
    def load(self):
        """ Loads the Transformed Data into File System and returns it 

        Returns
        -------
        transformed_df
            The dataset (Pandas Dataframe) equivalent of the file stored after transformations on source

        """
        if self.transformed_df is None:
            self.transform()

        self.transformed_df.to_csv(self._data_folder + 'forecasting_history.csv', index=False)  

    @staticmethod
    def age(born): 
        """ Computes Age given Date of Birth

        Parameters
        ----------
        born : str
            The date of birth in String and in YYYY-MM-DD format 

        Returns
        -------
        age
            The numeric age of the customer

        """
        born = datetime.strptime(born, "%Y-%m-%d").date() 
        today = date.today() 
        return today.year - born.year - ((today.month,  today.day) < (born.month,  born.day)) 

    def haversine_vectorize(self, lon1, lat1, lon2, lat2):
        """ Returns distance, in miles, between one set of longitude/latitude coordinates and another

        Parameters
        ----------
        lon1 : float
            Longitude of source
        lat1 : float
            Latitude of source
        lon2 : float
            Longitude of target
        lat2 : float
            Latitude of target

        Returns
        -------
        miles
            The distance between source and destination in miles

        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
 
        newlon = lon2 - lon1
        newlat = lat2 - lat1
 
        haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2
 
        dist = 2 * np.arcsin(np.sqrt(haver_formula ))
        miles = 3958 * dist #6367 for distance in KM 
        return miles

    def classify_distance(_haversine_distance): 
        if (int(_haversine_distance) < 20):
            return 'Low'
        elif (int(_haversine_distance) >= 50):
            return 'High'
        else:
            return 'Medium'

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
            transformed_df = pd.read_csv(self._data_folder + "transformed_data.csv")
        except Exception:
            print("Did not find the transformed_data.csv")
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
        source_df["txn_dt"] = pd.to_datetime(source_df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

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

        # Create derived attribute to indicate if the transaction is physical or on the internet
        source_df['is_internet'] = source_df.apply(lambda x: self.is_txn_internet(x['category']),axis=1)

        # Create derived attribute to reduce the categories into more bubbled up ones
        source_df['normalized_category'] = source_df.apply(lambda x: self.normalize_category(x['category']),axis=1)   

        # Drop the above columns
        cols_to_drop = ['cc_num','person','first','last','dob','sex','street','city','state','zip','person_loc',
                'merchant_loc','lat','long','trans_num','trans_date_trans_time','unix_time',
                'dob_dt','txn_dt','txn_hour','address','category','merch_lat','merch_long','normalized_job', 'merchant']
        trimmed_df = source_df.drop(columns=cols_to_drop,errors='ignore')

        # Drop the column with the serial number
        for col in trimmed_df.columns:
            idx = source_df.columns.get_loc(col)
            if idx == 0:
                trimmed_df = trimmed_df.drop(columns=source_df.columns[idx])

        # Get the class df
        class_df = trimmed_df[['is_fraud']].copy()

        # Apply scaling to all numeric columns
        fraud_numeric_features_df = self._apply_scaling(trimmed_df) 

        # Apply Encoding to categorical columns
        fraud_cat_features_df = self._apply_encoding(trimmed_df)

        # Generate Final features 
        self.transformed_df = pd.concat([fraud_cat_features_df, fraud_numeric_features_df, class_df], axis=1)
        return self.transformed_df

    def _apply_scaling(self, trimmed_df):
        """ Applies Scaling to the numeric data elements

        Parameters
        ----------
        trimmed_df : df
            The dataset after removing unneeded columns from source

        Returns
        -------
        fraud_numeric_features_df
            The dataset (Pandas Dataframe) containing the transformed numeric elements

        """
        # Min Max Scaler is a good choice for all these attributes
        min_max_scaler = preprocessing.MinMaxScaler()

        trimmed_df[['normalized_amt']] = min_max_scaler.fit_transform(trimmed_df[['amt']])
        trimmed_df[['normalized_city_pop']] = min_max_scaler.fit_transform(trimmed_df[['city_pop']])
        trimmed_df[['normalized_age']] = min_max_scaler.fit_transform(trimmed_df[['age']])
        trimmed_df[['normalized_distance_from_merchant']] = min_max_scaler.fit_transform(trimmed_df[['distance_from_merchant']])

        cols_to_drop = ['amt','city_pop','age','distance_from_merchant']
        trimmed_df = trimmed_df.drop(columns=cols_to_drop,errors='ignore')

        fraud_numeric_features_df = trimmed_df[['normalized_amt','normalized_city_pop','normalized_age','normalized_distance_from_merchant']].copy()
        return fraud_numeric_features_df

    def _apply_encoding(self, trimmed_df):
        """ Applies Encoding to the categorical data elements

        Parameters
        ----------
        trimmed_df : df
            The dataset after removing unneeded columns from source

        Returns
        -------
        fraud_cat_features_df
            The dataset (Pandas Dataframe) containing the transformed categorical elements

        """
        categorical_df = trimmed_df[['is_internet','normalized_category','job','txn_weekday','txn_month','part_of_day']].copy()

        # Apply one-hot encoder to category
        ohc_category = OneHotEncoder(categories=[['Entertainment','Home','Misc','Shopping']])
        ohe_category= ohc_category.fit_transform(categorical_df['normalized_category'].values.reshape(-1,1)).toarray()
        df_category = pd.DataFrame(ohe_category, columns = ohc_category.categories_[0])

        # Fix the order for Part of Day
        ordered_part_of_day = ['Late Night','Early Morning', 'Morning','Afternoon','Evening','Night']

        # Create the Ordinal Encoder
        oe_daypart = OrdinalEncoder(categories=[ordered_part_of_day])

        part_of_day_df = categorical_df[['part_of_day']].copy()
        part_of_day_df[['part_of_day']] = oe_daypart.fit_transform(part_of_day_df[['part_of_day']])

        # Fix the order for months
        ordered_month = ['January','February', 'March','April','May','June','July','August','September',
                      'October','November','December']

        # Create the Ordinal Encoder
        oe_month = OrdinalEncoder(categories=[ordered_month])

        month_df = categorical_df[['txn_month']].copy()
        month_df[['txn_month']] = oe_month.fit_transform(month_df[['txn_month']])

        # Fix the order for months
        ordered_weekday = ['Sunday','Monday', 'Tuesday','Wednesday','Thursday','Friday','Saturday']

        # Create the Ordinal Encoder
        oe_weekday = OrdinalEncoder(categories=[ordered_weekday])

        weekday_df = categorical_df[['txn_weekday']].copy()
        weekday_df[['txn_weekday']] = oe_weekday.fit_transform(weekday_df[['txn_weekday']])

        # Create instance of labelencoder
        labelencoder = LabelEncoder()

        job_df = categorical_df[['job']].copy()
        job_df['job_enc'] = labelencoder.fit_transform(job_df['job'])

        cols_to_drop = ['job']
        job_df = job_df.drop(columns=cols_to_drop,errors='ignore')

        # Concatenate all features and return
        fraud_cat_features_df = pd.concat([job_df, df_category, weekday_df, month_df, part_of_day_df], axis=1)
        return fraud_cat_features_df
    
    def load(self):
        """ Loads the Transformed Data into File System and returns it 

        Returns
        -------
        transformed_df
            The dataset (Pandas Dataframe) equivalent of the file stored after transformations on source

        """
        if self.transformed_df is None:
            self.transform()

        self.transformed_df.to_csv(self._data_folder + 'transformed_data.csv', index=False)  

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

    def is_txn_internet(self, _category): 
        """ Returns whether the transaction was made on the internet or was at physical location E.g. Point of Sale (POS)

        Parameters
        ----------
        _category : str
            Transaction Category

        Returns
        -------
        is_txn_internet
            1 if the Transaction was made on Internet otherwise 0

        """
        if (_category.endswith('_net')):
            return 1
        else:
            return 0

    def normalize_category(self, _category): 
        """ Returns a normalized category by clubbing some of the values together

        Parameters
        ----------
        _category : str
            Transaction Category

        Returns
        -------
        normalized_category: str
            One out of Shopping, Home, Entertainment and Misc

        """
        if ((_category.find('shopping_')!=-1) | (_category.find('grocery_')!=-1)):
            return 'Shopping'
        elif ((_category.find('personal_care')!=-1) | (_category.find('health_fitness')!=-1) 
            | (_category.find('home')!=-1) | (_category.find('kids_pets')!=-1)):
            return 'Home'
        elif ((_category.find('entertainment')!=-1) | (_category.find('food_dining')!=-1) | 
             (_category.find('gas_')!=-1) | (_category.find('travel')!=-1)):
            return 'Entertainment'
        else:
            return 'Misc'

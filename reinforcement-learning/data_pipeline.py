import numpy as np
import pandas as pd
from datetime import datetime, date 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

class ETL_Pipeline:
    """
    A class used to represent the Data Pipeline

    ...

    Attributes
    ----------
    _data_folder : str
        a string used to store the folder for the input and intermediate transformed data
    sent_df : df
        the dataframe representing the sent email data
    response_df : df
        the dataframe representing the responded email data
    customer_df : df
        the dataframe representing the customer data
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
        data_folder : str
            The folder with the sent, response and customer files are kept

        """
        self._data_folder = data_folder
        self.sent_df = None
        self.response_df = None
        self.customer_df = None
        self.transformed_df = None

        # Store encoding mappings
        self.customer_type_mappings = None
        self.gender_mappings = None
        self.email_domain_mappings = None
    
    def process(self, sent_file, response_file, customer_file):
        """ Executes the Pipeline to return transformed dataset 

        Parameters
        ----------
        sent_file : str
            the name of the file for sent emails
        response_file : str
            the name of the file for responded emails
        customer_file : str
            the name of the file for customer data    

        Returns
        -------
        transformed_df
            The dataset (Pandas Dataframe) equivalent of the transformed file

        """
        try:
            transformed_df = pd.read_csv(self._data_folder + "email_campaign_data.csv")
        except Exception:
            print("Did not find the email_campaign_data.csv")
            self.extract(sent_file, response_file, customer_file)
            self.transform()
            self.load()   
            transformed_df = self.transformed_df
        
        return transformed_df

    def extract(self, sent_file, response_file, customer_file):
        """ Reads the sent, response and customer files to return corresponding datasets 
        
        Returns
        -------
        sent_df
            The dataset (Pandas Dataframe) equivalent of the sent emails file
        response_df
            The dataset (Pandas Dataframe) equivalent of the responded emails file
        customer_df
            The dataset (Pandas Dataframe) equivalent of the customer data file            
        """
        self.sent_df = pd.read_csv(self._data_folder + sent_file)
        self.response_df = pd.read_csv(self._data_folder + response_file)
        self.customer_df = pd.read_csv(self._data_folder + customer_file)
        return (self.sent_df, self.response_df, self.customer_df)

    def transform(self):
        """ Transforms the source dataframes to return transformed dataset 
        
        Returns
        -------
        transformed_df
            The dataset (Pandas Dataframe) after performing transformations on the source data

        """
        # Drop duplicates since we are only interested if the customer responded (once or more than once is treated the same)
        df_received_copy = self.response_df.copy(deep=True)
        df_received_copy = df_received_copy.drop_duplicates(subset=['SubjectLine_ID','Customer_ID']) 

        # Merge or Join dataframes
        df_merge = pd.merge(self.sent_df, self.customer_df, how='left', on=['Customer_ID'])
        df_merge_recd = pd.merge(df_merge, df_received_copy, how='left', on=['SubjectLine_ID','Customer_ID'])

        # Find email domains
        domain_list = df_merge_recd['Email_Address'].str.split('@').str[1]
        df_merge_recd['Email_Domain'] = domain_list

        # Compute age group from Age
        df_merge_recd['Age_Group'] = df_merge_recd['Age'].apply(ETL_Pipeline.age_group) 

        # Compute age group from Age
        df_merge_recd['Tenure_Group'] = df_merge_recd['Tenure'].apply(ETL_Pipeline.tenure_group) 

        # Compute Response Received
        df_merge_recd[['Responded_Date']] = df_merge_recd[['Responded_Date']].fillna('NoResponseReceived')
        df_merge_recd['Response_Received'] = df_merge_recd['Responded_Date'].apply(ETL_Pipeline.response_received) 

        # Get week day from date
        df_merge_recd["sent_dt"] = pd.to_datetime(df_merge_recd['Sent_Date'], format='%Y-%m-%d', errors='coerce')
        df_merge_recd["Sent_Day"] = df_merge_recd['sent_dt'].dt.weekday

        # Label Encode Gender, Type and Domain
        gender_encoder = preprocessing.LabelEncoder() 
        type_encoder = preprocessing.LabelEncoder() 
        domain_encoder = preprocessing.LabelEncoder() 

        df_merge_recd['Gender']= gender_encoder.fit_transform(df_merge_recd['Gender']) 
        df_merge_recd['Type']= type_encoder.fit_transform(df_merge_recd['Type']) 
        df_merge_recd['Email_Domain']= domain_encoder.fit_transform(df_merge_recd['Email_Domain']) 

        self.email_domain_mappings = dict(zip(domain_encoder.transform(domain_encoder.classes_),domain_encoder.classes_))
        self.customer_type_mappings = dict(zip(type_encoder.transform(type_encoder.classes_),type_encoder.classes_))
        self.gender_mappings = dict(zip(gender_encoder.transform(gender_encoder.classes_),gender_encoder.classes_))

        #print(self.email_domain_mappings)
        #print(self.customer_type_mappings)
        #print(self.gender_mappings)

        # Drop columns
        cols_to_drop = ['Sent_Date','Customer_ID','Email_Address','Age','Tenure','Responded_Date','sent_dt']
        trimmed_df = df_merge_recd.drop(columns=cols_to_drop,errors='ignore')

        # Group the columns to reduce the amount of data processed
        grouped_multiple = trimmed_df.groupby(['SubjectLine_ID', 'Gender','Type','Email_Domain','Tenure_Group','Sent_Day']).agg({'Response_Received': ['sum','count']})
        grouped_multiple.columns = ['Response_Received','Sent_Emails']
        grouped_multiple = grouped_multiple.reset_index()

        self.transformed_df = trimmed_df

        return self.transformed_df
   
    def get_encoder_mappings(self):
        return (self.email_domain_mappings, self.gender_mappings, self.customer_type_mappings)
    
    def load(self):
        """ Loads the Transformed Data into File System and returns it 

        Returns
        -------
        transformed_df
            The dataset (Pandas Dataframe) equivalent of the file stored after transformations on source

        """
        if self.transformed_df is None:
            self.transform()

        self.transformed_df.to_csv(self._data_folder + 'email_campaign_data.csv', index=False)  

    @staticmethod
    def age_group(age): 
        if (age < 20):
            return 0
        elif (age >= 20 and age <25):
            return 1
        elif (age >= 25 and age <35):
            return 2
        elif (age >= 35 and age <45):
            return 3  
        else:
            return 4

    @staticmethod
    def tenure_group(tenure): 
        if (tenure < 5):
            return 0
        elif (tenure >= 5 and tenure <10):
            return 1
        elif (tenure >= 10 and tenure <15):
            return 2
        elif (tenure >= 15 and tenure <20):
            return 3  
        elif (tenure >= 20 and tenure <25):
            return 4  
        elif (tenure >= 25 and tenure <30):
            return 5 
        else:
            return 6
    
    @staticmethod
    def response_received(responded_date): 
        if responded_date == 'NoResponseReceived':
            return 0
        else:
            return 1
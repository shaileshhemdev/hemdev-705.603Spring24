import numpy as np
import pandas as pd
import json

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
            transformed_df = pd.read_csv(self._data_folder + "transformed_reviews.csv")
        except Exception:
            print("Did not find the transformed_reviews.csv")
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
        # Drop the columns we have concluded as not being meaningful
        cols_to_drop = ['user_id','asin','parent_asin','movie_title','subtitle','rating_number','price','bought_together',
                'store','images_y','videos','author','images_x','timestamp','features','average_rating']
        trimmed_df = source_df.drop(columns=cols_to_drop,errors='ignore')

        # Drop the column with the serial number
        for col in trimmed_df.columns:
            idx = source_df.columns.get_loc(col)
            if idx == 0:
                trimmed_df = trimmed_df.drop(columns=source_df.columns[idx])

        trimmed_df['categories'].fillna('', inplace=True)
        trimmed_df['categories'] = trimmed_df['categories'].apply(lambda x: ETL_Pipeline.parse_categories(x))

        trimmed_df['details'].fillna('', inplace=True)
        trimmed_df['tags'] = trimmed_df['details'].apply(lambda x: ETL_Pipeline.parse_details(x))

        # Drop the details
        cols_to_drop = ['details','description','verified_purchase']
        transformed_df = trimmed_df.drop(columns=cols_to_drop,errors='ignore')

        # Map ratings to sentiments
        sentiment_classes = {5 : 2, 4 : 2, 3 : 1, 2 : 0, 1 : 0} 
        transformed_df["class"] = transformed_df["rating"].map(sentiment_classes) 

        # Generate Final features 
        self.transformed_df = transformed_df
        return self.transformed_df

    @staticmethod
    def parse_categories(category):
        if (category == ''):
            return category
        else:
            try:
                start = category.index('[') + 1
                end   = category.index(']') 
                elems = category[start:end]
                result = elems.replace("'","")
                result = result.replace(","," ")
                return result
            except:
                return category

    @staticmethod
    def parse_details(details):
        if (details == ''):
            return details
        else:
            result = details.replace("'",'"')
            tags = ''
            try:
                res = json.loads(result)

                if 'Content advisory' in res.keys():
                    content_advisory = res["Content advisory"]
                    tags = tags + " ".join(content_advisory)
                
                if 'Genre' in res.keys():
                    genre = res["Genre"]
                    tags = tags + " ".join(genre)
            except:
                tags = ''
            
            return tags

    def load(self):
        """ Loads the Transformed Data into File System and returns it 

        Returns
        -------
        transformed_df
            The dataset (Pandas Dataframe) equivalent of the file stored after transformations on source

        """
        if self.transformed_df is None:
            self.transform()

        self.transformed_df.to_csv(self._data_folder + 'transformed_reviews.csv', index=False)  

 
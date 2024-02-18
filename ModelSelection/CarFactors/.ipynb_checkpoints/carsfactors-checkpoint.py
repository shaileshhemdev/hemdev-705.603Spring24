# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Regression Model and Metrics
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class carsfactors:
    def __init__(self):
        self.modelLearn = False
        self.stats = 0

    def model_learn(self):
        # Importing the dataset into a pandas dataframe
        dataset = pd.read_csv("cars.csv")

        # We need manufacturer_name, transmission, color, odometer_value, year_produced, engine_type, body_type, has_warranty, state, drivetrain, price_usd, number_of_photos, up_counter
        cols_to_drop = ['engine_fuel','engine_has_gas','engine_capacity','feature_0','feature_1','feature_2','feature_3',
               'feature_4','feature_5','feature_6','feature_7','feature_8','feature_9','is_exchangeable', 
                'location_region','model_name']
        
        # We will drop above columns (see readme for rationale)
        trimmed_df = dataset.drop(columns=cols_to_drop,errors='ignore')
 
        # Do the ordinal Encoder for car type to reflect that some cars are bigger than others.  
        # This is the order 'universal','hatchback', 'cabriolet','coupe','sedan','liftback', 'suv', 'minivan', 'van','pickup', 'minibus','limousine'
        # make sure this is the entire set by using unique()
        # create a seperate dataframe for the ordinal number - so you must strip it out and save the column
        # make sure to save the OrdinalEncoder for future encoding due to inference
        # Fix the order for body types
        ordered_body_types = ['universal','hatchback', 'cabriolet','coupe','sedan','liftback', 'suv', 
                      'minivan', 'van','pickup', 'minibus','limousine']
        self.oe = OrdinalEncoder(categories=[ordered_body_types])
        
        # Perform the transformation on a copy of the dataframe
        body_type_df = trimmed_df[['body_type']].copy()
        body_type_df[['body_type']] = self.oe.fit_transform(body_type_df[['body_type']])

        # Apply one-hot encoder to each column with categorical data starting with transmission
        self.ohc_transmission = OneHotEncoder()
        ohe_transmission = self.ohc_transmission.fit_transform(trimmed_df['transmission'].values.reshape(-1,1)).toarray()
        df_transmission = pd.DataFrame(ohe_transmission, columns = self.ohc_transmission.categories_[0])

        # Apply one-hot encoder to Manufacturer
        self.ohc_manufacturer = OneHotEncoder()
        ohe_manufacturer = self.ohc_manufacturer.fit_transform(trimmed_df['manufacturer_name'].values.reshape(-1,1)).toarray()
        df_manufacturer = pd.DataFrame(ohe_manufacturer, columns = self.ohc_manufacturer.categories_[0])

        # Apply one-hot encoder to color
        self.ohc_color = OneHotEncoder()
        ohe_color = self.ohc_color.fit_transform(trimmed_df['color'].values.reshape(-1,1)).toarray()
        df_color = pd.DataFrame(ohe_color, columns = self.ohc_color.categories_[0])

        # Apply one-hot encoder to engine type
        self.ohc_engine = OneHotEncoder()
        ohe_engine = self.ohc_engine.fit_transform(trimmed_df['engine_type'].values.reshape(-1,1)).toarray()
        df_engine = pd.DataFrame(ohe_engine, columns = self.ohc_engine.categories_[0])

        # Apply one-hot encoder to drive train 
        self.ohc_drivetrain = OneHotEncoder()
        ohe_drivetrain = self.ohc_drivetrain.fit_transform(trimmed_df['drivetrain'].values.reshape(-1,1)).toarray()
        df_drivetrain = pd.DataFrame(ohe_drivetrain, columns = self.ohc_drivetrain.categories_[0])

        # Apply one-hot encoder to state 
        self.ohc_state = OneHotEncoder()
        ohe_state = self.ohc_state.fit_transform(trimmed_df['state'].values.reshape(-1,1)).toarray()
        df_state = pd.DataFrame(ohe_state, columns = self.ohc_state.categories_[0])
    
        # We will use Label Encoder for has_warranty since I presume having it has higher order
        self.le_warranty = LabelEncoder()
        has_warranty_df = trimmed_df[['has_warranty']].copy()
        has_warranty_df['has_warranty'] = self.le_warranty.fit_transform(has_warranty_df['has_warranty']) 

        # Finally we will extract the other columns 
        numeric_df = trimmed_df[['odometer_value','year_produced','price_usd']].copy()

        # Create a minmaxscaler to scale all the values
        self.min_max_scaler = MinMaxScaler()
        normalizable_cols = ['odometer_value','year_produced','price_usd']
        numeric_df[normalizable_cols] = self.min_max_scaler.fit_transform(trimmed_df[normalizable_cols])
        
        # Concatenate all the dataframes (hence delete is not needed)
        car_factors_features = pd.concat([df_manufacturer, df_transmission, df_color, df_engine, df_drivetrain, df_state,
                                 body_type_df, has_warranty_df, numeric_df], axis=1)
  
        # Seperate X and y (features and label)  The last feature "duration_listed" is the label (y)
        X = car_factors_features.values 
        y = trimmed_df['duration_listed'].values
        
        # Splitting the dataset into the Training set and Test set 
        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.1,  random_state=0)
        
        # Select useful model to deal with regression (it is not categorical for the number of days can vary quite a bit)
        #SVC(kernel='rbf') This runs slow 
        self.model = LinearRegression() 
        self.pipe_lr = make_pipeline(self.model)
        self.pipe_lr.fit(X_train, y_train)
        
        self.stats = self.model.score(X_train, y_train)
        self.modelLearn = True

    # this demonstrates how you have to convert these values using the encoders and scalers above (if you choose these columns - you are free to choose any you like)
    def model_infer(self,manufacturer, transmission, color, engine, drivetrain, state, bodytype, has_warranty, 
                    odometer, year, price):
        if(self.modelLearn == False):
            self.model_learn()

        # Convert the manufacturer into a numpy array with the correct encoding
        carManufacturerTest = np.array([manufacturer])
        carHotManufacturerTest = self.ohc_manufacturer.transform(carManufacturerTest.reshape(-1,1)).toarray()
        
        # Convert the transmission into a numpy array with the correct encoding
        carTransmissionTest = np.array([transmission])
        carHotTransmissionTest = self.ohc_transmission.transform(carTransmissionTest.reshape(-1,1)).toarray()

        # Convert the color into a numpy array with the correct encoding
        carColorTest = np.array([color])
        carHotColorTest = self.ohc_color.transform(carColorTest.reshape(-1,1)).toarray()
        
        # Convert the engine into a numpy array with the correct encoding
        carEngineTest = np.array([engine])
        carHotEngineTest = self.ohc_engine.transform(carEngineTest.reshape(-1,1)).toarray()
        
        # Convert the drivertrain into a numpy array with the correct encoding
        carDriveTrainTest = np.array([drivetrain])
        carHotDriveTrainTest = self.ohc_drivetrain.transform(carDriveTrainTest.reshape(-1,1)).toarray()
        
        # Convert the state into a numpy array with the correct encoding
        carStateTest = np.array([state])
        carHotStateTest = self.ohc_state.transform(carStateTest.reshape(-1,1)).toarray()
        
        # Convert the body type into a numpy array that holds the correct encoding
        bodytypeTest = np.array([[bodytype]])
        bodytypeHotTest = self.oe.fit_transform(bodytypeTest)
        
        # Convert the has_warranty into a numpy array that holds the correct encoding
        hasWarrantyTest = np.array([[has_warranty]])
        hasWarrantyHotTest = self.le_warranty.fit_transform(hasWarrantyTest)
           
        # Add all of the above
        total = np.concatenate((carHotManufacturerTest,carHotTransmissionTest,carHotColorTest,carHotEngineTest,
                                carHotDriveTrainTest, carHotStateTest, bodytypeHotTest,np.array([hasWarrantyHotTest])), axis=1)
        
        # Add the numeric columns
        othercolumns = np.array([[odometer ,year, price]])
        otherHotcolumns = self.min_max_scaler.transform(othercolumns)
        
        # build a complete test array and then predict 
        totaltotal = np.concatenate((total, otherHotcolumns),1)

        #must scale
        attempt = 1
        
        #determine prediction
        y_pred = self.pipe_lr.predict(totaltotal)
        return str(y_pred)
        
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)

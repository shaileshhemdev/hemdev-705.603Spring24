import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score


class randomforest:
    def __init__(self):
        self.modelLearn = False
        self.stats = 0
    def model_learn(self):
        print('Model Learning')
        # Importing the dataset
        dataset = pd.read_csv('Social_Network_Ads.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values


        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler()
        X_train = self.sc.fit_transform(X_train)
        X_test = self.sc.transform(X_test)

        # Training the Random Forest Classification model on the Training set
        from sklearn.ensemble import RandomForestClassifier
        self.classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        self.classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = self.classifier.predict(X_test)
        self.stats = str(accuracy_score(y_test, y_pred))
        return True

    def model_infer(self, age,salary):
        if(self.modelLearn == False):
            #must learn first
            self.modelLearn = self.model_learn()
        if( self.classifier.predict(self.sc.transform([[age,salary]]))[0] == 1):
            determination = 'Yes, A good Candidate'
        else:
            determination = 'No, a Bad Candidate'
        return determination

    def model_stats(self):
        if(self.modelLearn == False):
            #must learn first
            self.modelLearn = self.model_learn()
        return self.stats

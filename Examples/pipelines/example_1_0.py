import sys
import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


current_dir = os.path.dirname(os.path.abspath(__name__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

project_root = Path(__name__).parent.parent
raw_data_file = os.path.join(str(project_root), "Examples/datasets", "adult_data", "adult_data.csv")

#####################
data = pd.read_csv(raw_data_file)

data['occupation'] = data['occupation'].str.lower() 
data['occupation'] = data['occupation'].str.replace('-', ' ') 

data['native-country'] = data['native-country'].apply(lambda x: 'North America')

# train test split
train_data, test_data = train_test_split( data, test_size=0.2)
X_train, y_train = train_data.drop('salary', axis=1), train_data['salary']
X_test, y_test = test_data.drop('salary', axis=1), test_data['salary']


numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns 
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('normalizer', Normalizer()) 
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    [
        ('num', numeric_transformer, numeric_features), 
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
#####################

import json
from sklearn.metrics import classification_report, accuracy_score

report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)

report["accuracy"] = accuracy

with open("Examples/Predicted/Aggregation_Errors0.json", "w") as json_file:
    json.dump(report, json_file, indent=4)

print("Prediction statistics saved to prediction_stats.json")

import os
import sys
from pathlib import Path
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__name__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

project_root = Path(__name__).parent.parent
raw_data_file = os.path.join(project_root, "Examples/datasets", "adult_data", "adult_data.csv")

#####################
data = pd.read_csv(raw_data_file)

def text_preprocessing(text_series):
    text_series = text_series.str.lower()
    text_series = text_series.str.replace('-', ' ')
    return text_series

def spatial_aggregation(location_series):
    location_series = location_series.apply(lambda x: 'North America' if x in ['United-States', 'Canada', 'Mexico'] else x)
    return location_series  

data['occupation'] = text_preprocessing(data['occupation'])
data['native-country'] = spatial_aggregation(data['native-country'])

# train test split
train_data, test_data = train_test_split( data, test_size=0.2)
X_train, y_train = train_data.drop('salary', axis=1), train_data['salary']
X_test, y_test = test_data.drop('salary', axis=1), test_data['salary']


numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

#####################

import json
from sklearn.metrics import classification_report, accuracy_score

report = classification_report(y_test, y_pred, output_dict=True)  # Output as a dictionary
accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy separately

# Add accuracy to the report
report["accuracy"] = accuracy

# Save the metrics to a JSON file
with open("Examples/Predicted/Aggregation_Errors1.json", "w") as json_file:
    json.dump(report, json_file, indent=4)

print("Prediction statistics saved to prediction_stats.json")

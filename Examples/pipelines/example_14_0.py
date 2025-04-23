import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

current_dir = os.path.dirname(os.path.abspath(__name__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

project_root = Path(__name__).parent.parent
raw_data_file = os.path.join(project_root, "Examples/datasets", "adult_data", "adult_data.csv")
#####################
data = pd.read_csv(raw_data_file)

X = data.drop(columns=['salary'])
y = data['salary']

categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
X_train, y_train = train_data.drop(columns=['salary']), train_data['salary']
X_test, y_test = test_data.drop(columns=['salary']), test_data['salary']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
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
with open("Examples/Predicted/Specification_Bias0.json", "w") as json_file:
    json.dump(report, json_file, indent=4)
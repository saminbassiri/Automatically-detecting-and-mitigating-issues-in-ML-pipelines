import pandas as pd
import os
import sys
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__name__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

project_root = Path(__name__).parent.parent

raw_data_file = os.path.join(str(project_root), "Examples/datasets", "adult_data", "adult_data.csv")

#####################
data = pd.read_csv(raw_data_file)

numeric_columns = ['age', 'hours-per-week']
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

X = data[numeric_columns + categorical_columns]
y = data['salary']

y_encoded = LabelEncoder().fit_transform(y)

data_combined = X.copy()
data_combined['salary'] = y_encoded

train, test = train_test_split(data_combined, test_size=0.2, random_state=42)
X_train, y_train = train.drop(columns=['salary']), train['salary']
X_test, y_test = test.drop(columns=['salary']), test['salary']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns), 
        ('cat', categorical_transformer, categorical_columns)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression()) 
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#####################

import json
from sklearn.metrics import classification_report, accuracy_score

report = classification_report(y_test, y_pred, output_dict=True)  # Output as a dictionary
accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy separately

# Add accuracy to the report
report["accuracy"] = accuracy

# Save the metrics to a JSON file
with open("Examples/Predicted/Data_Leakage1.json", "w") as json_file:
    json.dump(report, json_file, indent=4)
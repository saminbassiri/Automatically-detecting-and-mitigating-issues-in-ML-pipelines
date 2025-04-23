import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__name__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
project_root = Path(__name__).parent.parent
raw_data_file = os.path.join(project_root, "Examples/datasets", "adult_data", "adult_data.csv")

#####################
data = pd.read_csv(raw_data_file)

#protected_attribute: 'race', 'gender'

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42) 
X_train, y_train = train_data.drop('salary', axis=1), train_data['salary']
X_test, y_test = test_data.drop('salary', axis=1), test_data['salary']

proxy_attributes = set()

df_encoded = pd.get_dummies(X_train, drop_first=True)

corr_matrix = df_encoded.corr().abs()

for protected_attribute in ['race', 'gender']:
    if protected_attribute in X_train.columns:
        encoded_columns = [col for col in df_encoded.columns if col.startswith(protected_attribute)]
        for encoded_col in encoded_columns:
            high_corr_attributes = corr_matrix[encoded_col][corr_matrix[encoded_col] > 0.8].index.tolist()
            proxy_attributes.update(high_corr_attributes)

proxy_attributes = {col for col in proxy_attributes if not any(col.startswith(protected_attr) for protected_attr in ['race', 'gender'])}

proxy_attributes = list(proxy_attributes)

features_to_include = [col for col in X_train.columns if col not in ['race', 'gender'] + proxy_attributes]
X_train = X_train[features_to_include]
X_test = X_test[features_to_include]

categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'
)

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
with open("Examples/Predicted/Specification_Bias1.json", "w") as json_file:
    json.dump(report, json_file, indent=4)
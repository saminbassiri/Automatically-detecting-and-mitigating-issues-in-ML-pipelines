import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__name__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

project_root = Path(__name__).parent.parent

raw_data_file = os.path.join(project_root, "Examples/datasets", "diabetes_indicator", "5050_split.csv")
#####################
data = pd.read_csv(raw_data_file)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42) 
X_train, y_train = train_data.drop('Diabetes_binary', axis=1), train_data['Diabetes_binary']
X_test, y_test = test_data.drop('Diabetes_binary', axis=1), test_data['Diabetes_binary']


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train_resampled, y_train_resampled)

y_pred = pipeline.predict(X_test)

#####################

import json
from sklearn.metrics import classification_report, accuracy_score

report = classification_report(y_test, y_pred, output_dict=True)  # Output as a dictionary
accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy separately

# Add accuracy to the report
report["accuracy"] = accuracy

# Save the metrics to a JSON file
with open("Examples/Predicted/Preprocessing_Order1.json", "w") as json_file:
    json.dump(report, json_file, indent=4)

import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

current_dir = os.path.dirname(os.path.abspath(__name__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

project_root = Path(__name__).parent.parent

raw_data_file = os.path.join(project_root, "Examples/datasets", "diabetes_indicator", "binary_health_indicators.csv")

#####################
data = pd.read_csv(raw_data_file)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42 ) 
X_train, y_train = train_data.drop('Diabetes_binary', axis=1), train_data['Diabetes_binary']
X_test, y_test = test_data.drop('Diabetes_binary', axis=1), test_data['Diabetes_binary']

model = RandomForestClassifier(n_estimators=100, random_state=42)
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
with open("Examples/Predicted/Spurious_Correlations0.json", "w") as json_file:
    json.dump(report, json_file, indent=4)

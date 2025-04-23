import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

current_dir = os.path.dirname(os.path.abspath(__name__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

project_root = Path(__name__).parent.parent

raw_data_file = os.path.join(project_root, "Examples/datasets", "diabetes_indicator", "binary_health_indicators.csv")


#####################

data = pd.read_csv(raw_data_file)

# train test split
train_data, test_data = train_test_split(data,test_size=0.2, random_state=42)
X_train, y_train = train_data.drop('Income', axis=1), train_data['Income']
X_test, y_test = test_data.drop('Income', axis=1), test_data['Income']

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=5000).fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

#####################

import json
from sklearn.metrics import classification_report, accuracy_score

report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)

report["accuracy"] = accuracy

with open("Examples/Predicted/Class_Imbalance1.json", "w") as json_file:
    json.dump(report, json_file, indent=4)


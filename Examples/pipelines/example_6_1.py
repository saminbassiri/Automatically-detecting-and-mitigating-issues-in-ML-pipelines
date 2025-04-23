import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

current_dir = os.path.dirname(os.path.abspath(__name__))

parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

project_root = Path(__name__).parent.parent

raw_data_file = os.path.join(project_root, "Examples/datasets", "diabetes_indicator", "5050_split.csv")


#####################
data = pd.read_csv(raw_data_file)

# train test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
X_train, y_train = train_data.drop('Diabetes_binary', axis=1), train_data['Diabetes_binary']
X_test, y_test = test_data.drop('Diabetes_binary', axis=1), test_data['Diabetes_binary']

age_dist_train = X_train['Age'].value_counts(normalize=True).round(2)
X_train_filtered = X_train[X_train['Age'] > 4]

chol_dist_train = X_train['HighChol'].value_counts(normalize=True).round(2)
X_train_filtered = X_train_filtered[X_train_filtered['HighChol'] > 0]

y_train_filtered = y_train.loc[X_train_filtered.index]

age_dist_test = X_test['Age'].value_counts(normalize=True).round(2)
X_test_filtered = X_test[X_test['Age'] > 4]

chol_dist_test = X_test['HighChol'].value_counts(normalize=True).round(2)
X_test_filtered = X_test_filtered[X_test_filtered['HighChol'] > 0]

y_test_filtered = y_test.loc[X_test_filtered.index]

if( not X_train_filtered.select_dtypes(include=['object']).empty):
    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train_filtered.select_dtypes(include=['object']))
    X_test_encoded = encoder.transform(X_test_filtered.select_dtypes(include=['object']))
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(X_train_filtered.select_dtypes(include=['object']).columns))
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(X_test_filtered.select_dtypes(include=['object']).columns))
else:
    X_train_encoded_df = X_train_filtered
    X_test_encoded_df = X_test_filtered

X_train_final = pd.concat([X_train_filtered.select_dtypes(exclude=['object']).reset_index(drop=True), X_train_encoded_df.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_filtered.select_dtypes(exclude=['object']).reset_index(drop=True), X_test_encoded_df.reset_index(drop=True)], axis=1)

X_train_final.columns = X_train_final.columns.astype(str)
X_test_final.columns = X_test_final.columns.astype(str)

model = LogisticRegression(max_iter=1000).fit(X_train_final, y_train_filtered)

y_pred = model.predict(X_test_final)

#####################

import json
from sklearn.metrics import classification_report, accuracy_score

report = classification_report(y_test_filtered, y_pred, output_dict=True)
accuracy = accuracy_score(y_test_filtered, y_pred)

report["accuracy"] = accuracy

with open("Examples/Predicted/Data_Filtering1.json", "w") as json_file:
    json.dump(report, json_file, indent=4)

# Automatically Detecting and Mitigating Issues in ML Pipelines  

This repository provides a Python library for **automatically detecting issues in machine learning (ML) pipelines** using a combination of **ML-Inspect** and **Large Language Models (LLMs)**. The system extracts execution details, intermediate data statistics, and model performance insights to enhance the detection of common ML pipeline issues.  

## Project Overview  
The project focuses on identifying and mitigating **eight common ML pipeline issues** by leveraging **ML-Inspect** to generate Directed Acyclic Graphs (DAGs) and using **LLMs** to analyze the extracted information. This approach enhances issue detection accuracy, improving ML model reliability and fairness.  

## Repository Structure  

### 1. `examples/`  
This directory contains **sample ML pipelines and corresponding datasets**. Some examples are **modified** to handle unsupported functions in ML-Inspect. These sample pipelines include both **correct** and **incorrect** implementations to evaluate issue detection. The `meta.json`  contains information about the issues and it is used to map an issue to the related python code.
`Predicted/`: Contains information from classification_report and accuracy_score. this is computed when running the pipeline.

### 2. `DATA/`  
Stores **resulting DAGs and intermediate information** for the example ML pipelines. The results are structured based on issue names:  
- `1` for **correct pipelines**  
- `0` for **incorrect pipelines**  

### 3. `PipelineDataGenerator.py`  
A core class that runs **ML-Inspect** on an input Python script to:  
- **Generate DAGs** that visualize ML pipeline execution steps.  
- **Extract intermediate results** for each processing step.
- **Use `DataAggregation.py`** to compute statistical summaries.  
- **Save DAGs** and related data for issue detection.  
- **Provide model-specific insights** based on the input ML pipeline.  

### 4. `DataAggregation.py`  
This module includes **aggregation functions** applied to **intermediate results** to extract statistical information. Key features include:  
- **Summary Statistics:** Mean, standard deviation, min/max, quartiles.  
- **Data Distribution Analysis:** Skewness, kurtosis for numerical features.  
- **Categorical Feature Analysis:** Unique values, frequent categories, missing values.  
- **Missing Value Summary:** Percentage and count of missing values.  

### 5. `LLM/`  
This folder contains scripts for interacting with **LLMs** to analyze ML pipeline issues:  
- **`API.py`** – Sends extracted pipeline data and prompts to an LLM for issue detection.  
- **`get_prompt.py`** – Handles **prompt engineering** for different types of ML pipeline issues.  

### 6. `Results/`  
Stores **LLM-generated results** based on the **DATA** directory and the ML pipeline examples. The results include:  
- **Detected issues** based on raw code, DAGs, and aggregated data.  
- **Issue explanations and potential fixes** suggested by the LLM.

### 6. `mlinspect/` 
- mlinspect the Python library with newly added functionalities for sklearn
- These new functions need to be revised later to make sure about their functionality by adding tests for them.

## How to Use  

1. **Prepare an ML pipeline script** and place it inside `examples/` update the `meta.json` accordingly.  
2. **Run `PipelineDataGenerator.py`** to generate a DAG and extract intermediate results.  
4. **Send extracted information to the LLM** using `LLM/API.py`.  
5. **Analyze the results** stored in the `Results/` directory.  

## Dependencies  
- Python 3.x  
- ML-Inspect  
- OpenAI API (for LLM interactions)  
- Pandas, NumPy, Scikit-learn  

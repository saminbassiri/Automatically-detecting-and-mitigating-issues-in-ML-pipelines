import json

def get_prompt1(PythonCode, description, issue, possible_solver):
    list_solver = "Possible fixes for this issue:"
    for fix_text in possible_solver:
        list_solver = list_solver + "\n\t" + fix_text
    
    prompt = f"""
    Analyze the following ML pipeline Python code for potential {issue} issue. 

    Python Code:
    ```
    {PythonCode}
    ```
    """ + """
    
    Return in JSON format (Return ONLY JSON, without code fences or additional text):
    {
        issue_detected: 0 or 1,
        more_info_required: 0 or 1,  \\1 If the issue cannot be fully determined from code alone (e.g., requires data-related insights) otherwise 0
        problematic_operators: [operator1, operator2], \\Empty if no issues,
        explanation: , \\Brief explanation of how the operators cause issue
    }
    
    """+ f"""
    Context:
    {description}
    {list_solver}

    Rules:
    1. Focus on "{issue}" issues.
    2. If the issue CANNOT be FULLy determined from code alone (e.g., requires data-related insights), state this clearly and request additional information.
    3. Provide specific, actionable explanations if issues are found.
    4. Return ONLY the JSON response, without extra text.
    5. Ensure all detected issues are backed by evidence from the code.
    """
    return prompt

def get_prompt2(PythonCode, DAG_json, description, issue, possible_solver):
    list_solver = "Possible fixes for this issue:"
    for fix_text in possible_solver:
        list_solver = list_solver + "\n\t" + fix_text
        
    prompt = f"""
    Analyze the following ML pipeline Python code and its corresponding DAG representation for potential {issue} issue. 
    The DAG (in JSON format) represents the execution order of the Python code:
    - Nodes: Each preprocessing operation
    - Edges: Data flow between operations (check to_neighbors field in node)
    - Format: In each node, "to_neighbors": [X] indicates an edge from this_node to X

    Python Code:
    ```
    {PythonCode}
    ```
    DAG Representation (Directed Acyclic Graph of operations):
    ```
    {json.dumps(DAG_json, indent=4)}
    ```
    """ + """
    
    Return in JSON format (Return ONLY JSON, without code fences or additional text):
    {
        issue_detected: 0 or 1,
        more_info_required: 0 or 1,  \\1 If the issue cannot be fully determined from code and DAG alone (e.g., requires data-related insights) otherwise 0
        problematic_operators: [operator1, operator2], \\Empty if no issues,
        explanation: , \\Brief explanation of how the operators cause issue
    }
    
    """ + f"""
    Context:
    {description}

    Rules:
    1. Strictly focus on possible "{issue}" issues.
    2. If the issue CANNOT be FULLy determined from code/DAG alone (e.g., requires data-related insights), state this clearly and request additional information.
    3. Provide specific, actionable explanations if issues are found.
    4. Cross-reference findings between the code and DAG.
    5. Return ONLY the JSON response, without extra text.
    6. Ensure all detected issues are backed by evidence from the code/DAG.
    """
    return prompt

def get_prompt3(PythonCode, intermediate_result, description, issue, possible_solver):
    list_solver = "Possible fixes for this issue:"
    for fix_text in possible_solver:
        list_solver = list_solver + "\n\t" + fix_text
        
    prompt = f"""
    Analyze the following ML pipeline Python code and its corresponding intermediate_result for potential {issue} issue in final result. 
    The intermediate_result (in JSON format) represents the aggregation of result for some operator in the Python code, each node shows:
    - Operator information (function, description for the operator in python code)
    - Statistical measures (count, mean, std per column for the operator result)
    - Distribution metrics (skewness, kurtosis per column for the operator result)
    - categorical feature analysis (counting unique values and most frequent category)
    - missing value summary (Summarize missing values for all features.)
    Prediction statistics: final model statistics base on predicted values and test data (not related to specific nodes)
    
    Python Code:
    ```
    {PythonCode}
    ```
    Intermediate_result (in JSON format):
    ```
    {json.dumps(intermediate_result, indent=4)}
    ```
    """ + """
    
    Return in JSON format (Return ONLY JSON, without code fences or additional text):
    {
        issue_detected_in_data: 0 or 1, \\1 if base on intermediate result, there is """ + f"{issue}" + """ issue in data
        problem_resolved_in_code: 0 or 1, \\ 1 If there is an issue but fixes applied, otherwise 0
        more_info_required: 0 or 1,  \\1 if statistical measures from other operators outside intermediate result is required
        problematic_operators: [operator1, operator2], \\Empty if no issues,
        explanation: , \\Brief explanation of how the operators cause issue
    }
    
    """ + f"""
    Context:
    {description}
    {list_solver}

    Rules:
    1. Strictly focus on possible "{issue}" issues.
    2. The intermediate results do not cover all operators.
    3. If the issue can't be fully determined from code or a provided intermediate result, say so and ask for more information.
    4. For each issue, explain how the intermediate result was used to detect issue. 
    5. The code could be fixed to resolve the issues. If you found a problem with the data, check the code to see if a solution was applied to resolve the issue.
    6. Provide specific, actionable explanations if issues are found.
    7. Return ONLY the JSON response, without extra text.
    8. Ensure all detected issues are backed by evidence from the code or intermediate results aggregation.
    """
    # 7. The DAG doesn't support all the Python functions, and they have a negative node ID. Use code to replace these functions or ignore them.
    return prompt
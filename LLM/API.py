import openai
import os
import json
from pathlib import Path
from LLM.get_prompt import *

class LLMdetector():
    
    def __init__(self):
        self.root = Path(__file__).parent
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.result_dict = {}
        self.result_count = 0
    
    def resetResult(self):
        self.result_dict = {}
    
    def read_python_file(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().split("#####################")[1]
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found."
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def analyze_code_with_gpt(self, prompt: str):
        try:
            messages=[
                    {"role": "developer", "content": "You are an AI assistant that reviews and analyzes Python code for possible issues."},
                    {"role": "user", "content": prompt}
                ]
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages = messages,
                temperature=0.2
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error during API call: {str(e)}"

    # Example usage
    def run_experiment(self, experiment_index, issue_name, code_type, python_code, issue_desc, DAG_json = None, issue_fix = None, Intermediate_result= None):
        code_content = self.read_python_file(python_code)
        if experiment_index == "1":
            prompt_text = get_prompt1(code_content, issue_desc, issue_name, issue_fix)
        
        elif experiment_index == "2" and DAG_json: 
            prompt_text = get_prompt2(code_content, DAG_json, issue_desc, issue_name, issue_fix)
        
        elif experiment_index == "3" and Intermediate_result: 
            prompt_text = get_prompt3(code_content, Intermediate_result, issue_desc, issue_name, issue_fix)
        
        result = self.analyze_code_with_gpt(prompt_text)
        self.result_count += 1
        self.result_dict.update({issue_name + str(code_type): result})
        
    
    def writeResult(self, path, fileName):
        # Save results as JSON file
        with open(path + fileName, "w") as json_file:
            json.dump(self.result_dict, json_file, indent=4)
        
        print(f"LLM result has been saved to {fileName}.json")

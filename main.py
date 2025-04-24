from PipelineDataGenerator import *
from Examples.get_example import *
from LLM.API import *
import random

def readJson(file_path):
    with open(file_path) as f:
        return json.load(f)
    
def RunPipelineAndGetDAG(pipelines, numPipe, file_name, path1, path2):
    runner = PipelineDataGenerator(pipelines[numPipe]["path"], numMaterializedRow=1000, dataSize=0)
    runner.getPipelineDag()
    runner.writeResult(path1,"DAG_info_" + file_name)
    runner.resetResult()
    runner.getIntermediateResultAgg(file_name)
    runner.writeResult(path2, "intermediate_info_" + file_name)

if __name__ == "__main__":
    pipelines, descriptions = GetExample().getData()
    path1 = "DATA/DAG_info/"
    path2 = "DATA/intermediate_info/"
    
    os.makedirs(path1, exist_ok=True)
    os.makedirs(path2, exist_ok=True)
    LLM = LLMdetector()
    
    # shuffle list of issues
    keys =  list(pipelines.keys())
    random.shuffle(keys)
    
    for numPipe in keys:
        issue = pipelines[numPipe]["error"]
        file_name = issue + str(pipelines[numPipe]["is_correct"]) + ".json"
        RunPipelineAndGetDAG(pipelines, numPipe, file_name, path1, path2)
    for experiment_num in ["1", "2", "3"]:
        for numPipe in keys:
            issue = pipelines[numPipe]["error"]
            file_name = issue + str(pipelines[numPipe]["is_correct"]) + ".json"
            LLM.run_experiment(experiment_num, descriptions[issue]["name"], 
                                pipelines[numPipe]["is_correct"], 
                                pipelines[numPipe]["path"],  
                                descriptions[issue]["text"],
                                DAG_json= readJson(path1 + "DAG_info_" + file_name),
                                issue_fix = descriptions[issue]["fix"],
                                Intermediate_result = readJson(path2 + "intermediate_info_" + file_name)
                                )
        LLM.writeResult("Result/LLM", "result"+ experiment_num +".json")
        LLM.resetResult()

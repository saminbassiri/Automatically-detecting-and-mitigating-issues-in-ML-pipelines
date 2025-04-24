import json
import pandas
from pandas import DataFrame
import networkx

from DataAggregation import *

from mlinspect._pipeline_inspector import PipelineInspector, InspectorResult
from mlinspect.inspections._materialize_first_output_rows import MaterializeFirstOutputRows
from mlinspect.inspections._inspection_result import InspectionResult
from mlinspect.inspections._inspection_input import InspectionRowDataSource

class PipelineDataGenerator:
    
    def __init__(self, inputPipeline, numMaterializedRow = None, dataSize = 0):
        self.inputPipeline = inputPipeline
        self.numMaterializedRow = numMaterializedRow
        self.dataSize = dataSize
        self.dag_info = {"nodes": {}}
        self.inspector_result = None
    
    def runPipeline(self):
        if(self.inspector_result is None):
            print("run pipeline")
            inspections_list = []
            if(self.numMaterializedRow):
                inspections_list.append(MaterializeFirstOutputRows(self.numMaterializedRow))
            if(len(inspections_list) > 0):
                self.inspector_result = PipelineInspector \
                .on_pipeline_from_py_file(self.inputPipeline) \
                .add_required_inspections(inspections_list)  \
                .execute()
            else:
                self.inspector_result = PipelineInspector \
                .on_pipeline_from_py_file(self.inputPipeline) \
                .execute()
        else:
            print("Pipeline has already been executed")
        
    def writeResult(self, path, fileName):
        # Save DAG info as JSON file
        with open(path + fileName, "w") as json_file:
            json.dump(self.dag_info, json_file, indent=4)
        
        print(f"DAG information has been saved to {fileName}.json")
    
    def resetResult(self):
        self.dag_info = {"nodes": {}}
            
    def getPipelineDag(self):
        self.runPipeline()
        dag_node_to_inspection_results = list(self.inspector_result.dag_node_to_inspection_results.items())
        print("iterate over DAG")
        for node, result in dag_node_to_inspection_results:
            node_data = {
                "node_id": node.node_id,
                "operator": node.operator_info.operator.value,
                "function": node.operator_info.function_info.function_name if node.operator_info.function_info else None,
                "description": node.details.description,
                "to_neighbors": [],
                # "code": node.optional_code_info.source_code if node.optional_code_info else None
            }
            if node_data["operator"] in ["ColumnTransformer", "Transformer"]:
                node_data.update({"columns": node.details.columns})
            nodeInDAG = self.dag_info["nodes"].get(node.node_id)
            if nodeInDAG is None:
                self.dag_info["nodes"][node.node_id] = node_data
            else:
                self.dag_info["nodes"][node.node_id].update(node_data)
        
        # Extracting edges from the DAG
        for edge in self.inspector_result.dag.edges:
            start_node = self.dag_info["nodes"].get(edge[0].node_id)
            if start_node is not None:
                start_node["to_neighbors"].append(edge[1].node_id)
    
    def readJson(self, file_path):
        with open(file_path) as f:
            return json.load(f)
    
    def getIntermediateResultAgg(self, file_name):
        self.runPipeline()
        dag_node_to_inspection_results = list(self.inspector_result.dag_node_to_inspection_results.items())
        print("aggregate intermediate result and add model info")
        self.dag_info["Prediction_statistics"]= self.readJson("Examples/Predicted/" + file_name)
        for node, result in dag_node_to_inspection_results:
            output_sample = result.get(MaterializeFirstOutputRows(self.numMaterializedRow), None)
            if output_sample is not None:
                if isinstance(output_sample, (DataFrame, np.ndarray)):
                    output_sample = output_sample
                else:
                    output_sample = None
            
            node_data = {
                "description": node.details.description,
                "function": node.operator_info.function_info.function_name if node.operator_info.function_info else None,
                "operator": node.operator_info.operator.value,
                }
            if output_sample is not None and (node_data["operator"] in ["Data Source", "Transformer", "predict"
                                                                        "Train Test Split", "Selection", 
                                                                        "Train Data", "Train Labels", "Estimator"] or node_data["function"] in ["ColumnTransformer"]):
                if has_numeric_value(output_sample):
                    node_data.update({
                        "summary_statistics": summary_statistics(output_sample),
                        "data_distribution": data_distribution(output_sample),
                        "categorical_feature_analysis": categorical_feature_analysis(output_sample),
                        "missing_value_summary": missing_value_summary(output_sample)
                    })
                    self.dag_info["nodes"][node.node_id] = json.loads(json.dumps(node_data, default=convert_numpy))
            
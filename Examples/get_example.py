import os
import json

from pathlib import Path

class GetExample:
    def __init__(self):
        self.namePrefix = "example_"
        self.dataFileName = "pipelines"
        self.root = Path(__file__).parent
        with open(str(self.root) + '/meta.json') as f:
            self.meta = json.load(f)

    def getData(self):
        result = {}
        for num in self.meta["file_index"]:
            # 0 for wrong and 1 for fixed
            result[num + "0"] = {
                "path": os.path.join(str(self.root), self.dataFileName, "example_" + num + "_0.py"),
                "error": self.meta["file_index"][num],
                "is_correct": 0
                }
            result[num + "1"] = {
                "path": os.path.join(str(self.root), self.dataFileName, "example_" + num + "_1.py"),
                "error": self.meta["file_index"][num],
                "is_correct": 1
                }
        return result, self.meta["issue_description"]


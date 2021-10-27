import json
import pandas as pd

def export_json_to_csv(file, result='results.csv'):
    with open(file) as data_file:
        data = json.load(data_file)
        df = pd.json_normalize(data, 'benchmarks')
        df.to_csv(result)

export_json_to_csv("../benchmark_results.json")
export_json_to_csv("../kapil_benchmark_results.json", 'kapil_results.csv')


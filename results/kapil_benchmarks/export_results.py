import json
import pandas as pd

with open(f"../../benchmark_results.json") as data_file:
    data = json.load(data_file)
df = pd.json_normalize(data, 'benchmarks')
df.to_csv(f"results.csv")


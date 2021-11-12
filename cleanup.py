import json
import sys

in_file_path = "results.json" if len(sys.argv) < 2 else sys.argv[1]
out_file_path = "cleaned_results.json" if len(sys.argv) < 3 else sys.argv[2]

with open(in_file_path, 'r', encoding='utf-8') as in_file:
    # load json
    raw_json = json.load(in_file)

    # filter out duplicates, keeping best datapoint
    methods = {}
    for new_dp in raw_json["benchmarks"]:
        name = new_dp["name"]

        if name not in methods or float(methods[name]["cpu_time"]) > float(new_dp["cpu_time"]):
            methods[name] = new_dp

    # write out results
    raw_json["benchmarks"] = [methods[m] for m in sorted(methods)]
    with open(out_file_path, 'w', encoding='utf-8') as out_file:
        json.dump(raw_json, out_file, indent=2)

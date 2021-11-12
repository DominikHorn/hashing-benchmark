import json
import os

with open('results.json') as results_file:
    data = json.load(results_file)
    existing_dp = [dp['name']for dp in data['benchmarks']]

    all_dp = {dp: True for dp in os.popen('cmake-build-release/src/benchmarks --benchmark_list_tests').read().strip().split('\n')}

    for dp in existing_dp:
        all_dp[dp] = False

    new_dp = [dp[0] for dp in all_dp.items() if dp[1]]

    # verify above is correct
    for dp in existing_dp:
        assert(dp not in new_dp)
        assert(dp in all_dp)

    print('|'.join(new_dp))

import os
import numpy as np
import sys
import subprocess

data = []
for num in np.linspace(1500, 150000, 100):
    for n_t in range(1, 10):
        tmp_data = []
        for _ in range(100):
            result = subprocess.run(f"./average_random_cuda {num} {2**n_t}", shell=True, check=True, capture_output=True, text=True).stdout
            init_time = float(result.split(",")[0])
            ctog_time = float(result.split(",")[1])
            add_time  = float(result.split(",")[2])
            gtoc_time = float(result.split(",")[3])
            sum_time  = float(result.split(",")[4])
            calc_time = ctog_time + add_time + gtoc_time + sum_time
            tmp_data.append([ init_time, calc_time ])
        tmp_data = np.array(tmp_data)
        data.append([
            num,
            2**n_t,
            np.mean(tmp_data[:, 0]),
            np.std(tmp_data[:, 0]),
            np.mean(tmp_data[:, 1]),
            np.std(tmp_data[:, 1]),
        ])
    print(f"finish: {num}")

np.savetxt('multi.csv', np.array(data), delimiter=',')
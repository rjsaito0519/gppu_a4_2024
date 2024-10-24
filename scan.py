import os
import numpy as np
import sys
import subprocess

data = []
for num in np.linspace(1500, 150000, 100):
    tmp_data = []
    for _ in range(100):
        result = subprocess.run(f"./average_random {num}", shell=True, check=True, capture_output=True, text=True).stdout
        init_time = float(result.split(",")[0])
        calc_time = float(result.split(",")[1])
        tmp_data.append([ init_time, calc_time ])
    tmp_data = np.array(tmp_data)
    data.append([
        num,
        np.mean(tmp_data[:, 0]),
         np.std(tmp_data[:, 0]),
        np.mean(tmp_data[:, 1]),
         np.std(tmp_data[:, 1]),
    ])
    print(f"finish: {num}")

np.savetxt('single.csv', np.array(data), delimiter=',')
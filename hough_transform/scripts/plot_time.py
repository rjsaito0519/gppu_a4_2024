import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors
import scipy.stats as stats
import matplotlib.patches as patches
from functools import reduce
import sys
import subprocess
import os
import matplotlib.ticker as ptick
import cv2
import copy
import uproot
import lmfit as lf
import lmfit.models as lfm
import statistics
import uncertainties

plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 28
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.axisbelow'] = True
plt.rcParams["xtick.direction"] = "in"               #x軸の目盛線を内向きへ
plt.rcParams["ytick.direction"] = "in"               #y軸の目盛線を内向きへ
plt.rcParams["xtick.minor.visible"] = True           #x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = True           #y軸補助目盛りの追加
plt.rcParams["xtick.major.size"] = 10                #x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 10                #y軸主目盛り線の長さ
plt.rcParams["xtick.minor.size"] = 5                 #x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 5                 #y軸補助目盛り線の長さ



result = []


file = uproot.open(f"../results/cuda_008.root")
tree = file["tree"].arrays(library="np")

n_duration = len(tree["duration"][0])

data = []
prev_evnum = -1
for evnum, duration in zip(tree["evnum"], tree["duration"]):
    if evnum != prev_evnum:
        data.append(duration)
        prev_evnum = evnum
    else:
        pass

data = np.array(data)

tmp_result = []
for i in range(n_duration):
    mean  = statistics.mean(data[:, i]*10**-6)
    stdev = statistics.stdev(data[:, i]*10**-6)
    
    time = uncertainties.ufloat(mean, stdev)

    print(f"{time:.2u} [ms]")
    tmp_result.append(mean)
result.append(tmp_result)

file = uproot.open(f"../results/cuda_008_omp.root")
tree = file["tree"].arrays(library="np")

n_duration = len(tree["duration"][0])

data = []
prev_evnum = -1
for evnum, duration in zip(tree["evnum"], tree["duration"]):
    if evnum != prev_evnum:
        data.append(duration)
        prev_evnum = evnum
    else:
        pass

data = np.array(data)

tmp_result = []
for i in range(n_duration):
    mean  = statistics.mean(data[:, i]*10**-6)
    stdev = statistics.stdev(data[:, i]*10**-6)
    
    time = uncertainties.ufloat(mean, stdev)

    print(f"{time:.2u} [ms]")
    tmp_result.append(mean)
result.append(tmp_result)

print(result)


sys.exit()

result = []
for pow in range(1, 10):
    n_thread = 2**pow
    print(n_thread)
    # file = uproot.open(f"../results/cpu.root")
    file = uproot.open(f"../results/cuda_{n_thread:0=3}.root")
    tree = file["tree"].arrays(library="np")

    n_duration = len(tree["duration"][0])

    data = []
    prev_evnum = -1
    for evnum, duration in zip(tree["evnum"], tree["duration"]):
        if evnum != prev_evnum:
            data.append(duration)
            prev_evnum = evnum
        else:
            pass

    data = np.array(data)

    tmp_result = [n_thread]
    for i in range(n_duration):
        mean  = statistics.mean(data[:, i]*10**-6)
        stdev = statistics.stdev(data[:, i]*10**-6)
        
        time = uncertainties.ufloat(mean, stdev)

        print(f"{time:.2u} [ms]")
        tmp_result.append(mean)
    result.append(tmp_result)

result = np.array(result)
label = [
    "prepare data",
    "cudaMalloc",
    "cudaMemcpy",
    "houghTransform",
    "cudaMemcpy",
    "std::max_element",
    "event selection",
    "total"
]

fig = plt.figure(figsize=(10, 6))
ax  = fig.add_subplot(111)
for i in range(len(result[0])-1):
    ax.plot(result[:, 0], result[:, i+1], "--o", label = label[i])
ax.set_yscale("log")
ax.set_xlabel("# of threads")
ax.set_ylabel("time [ms]")
ax.legend(loc='upper left', fontsize = 18, bbox_to_anchor=(1.0, 1))
plt.subplots_adjust(left = 0.13, right = 0.7, top = 0.98, bottom = 0.15)
plt.savefig("./cuda_n_thread.png", dpi=600, transparent=True)
plt.show()



fig = plt.figure(figsize=(10, 6))
ax  = fig.add_subplot(111)
for i in range(len(result[0])-1):
    ax.plot(result[:, 0], result[:, i+1], "--o", label = label[i])
ax.set_yscale("log")
ax.set_xlabel("# of threads")
ax.set_ylabel("time [ms]")
ax.set_ylim(0.1, 0.3)

# 指定した目盛りのみ表示
custom_ticks = [0.1, 0.2, 0.3]  # 表示したいy軸の目盛り
ax.set_yticks(custom_ticks)  # y軸の目盛りを指定
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())  # 指定目盛りの数値ラベルのみ表示

ax.legend(loc='upper left', fontsize = 18, bbox_to_anchor=(1.0, 1))
plt.subplots_adjust(left = 0.13, right = 0.7, top = 0.98, bottom = 0.15)
plt.savefig("./cuda_n_thread_zoom.png", dpi=600, transparent=True)
plt.show()

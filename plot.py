import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
import matplotlib.ticker as ptick
import lmfit as lf
import lmfit.models as lfm

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
plt.rcParams['figure.subplot.left'] = 0.15
plt.rcParams['figure.subplot.right'] = 0.98
plt.rcParams['figure.subplot.top'] = 0.98
plt.rcParams['figure.subplot.bottom'] = 0.12


data = np.genfromtxt("single.csv", delimiter=",")
num = np.linspace(1500, 150000, 100)

fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

model1 = lfm.LinearModel()
params = model1.guess(x = num, data = data[:, 0] )
result = model1.fit(x = num, data = data[:, 0], weights = 1/data[:, 1], params=params, method='leastsq')
print(result.fit_report())
fit_x = np.linspace(1500, 150000, 2)
fit_y = result.eval_components(x=fit_x)["linear"]

ax1.errorbar(num, data[:, 0], yerr=data[:, 1], fmt = "s", capsize = 0, markeredgecolor = "k", ms = 5, ecolor='k', color = "C0", zorder = 5, elinewidth = 1., label = r"$N_{K}$")
ax1.plot(fit_x, fit_y, color = "C2")

ax2.errorbar(num, data[:, 2], yerr=data[:, 3], fmt = "s", capsize = 0, markeredgecolor = "k", ms = 5, ecolor='k', color = "C1", zorder = 5, elinewidth = 1., label = r"$N_{K}$")

plt.subplots_adjust(left = 0.17, right = 0.98, top = 0.95, bottom = 0.12)
# plt.savefig(f"./img/diff_cross_sec_statistics_comparison_explain.png", dpi=600, transparent=True)
plt.show()

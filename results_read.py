import csv
import os
import matplotlib.pyplot as plt
import numpy as np

# fig = plt.figure()
# ax = fig.add_axes([0, 0, 1, 1])
modes = []
fps = []
bar_colors = []

# TODO create matplotlib table

perf_folder = 'perf'
subprocess_types = os.listdir(perf_folder)
COLORS = ['tab:orange', 'tab:blue', 'tab:orange']
bar_labels = []
colors_proxy = {}

for idx, subprocess_type in enumerate(subprocess_types):
    n_run_path = os.path.join(perf_folder, subprocess_type)
    n_envs_list = os.listdir(n_run_path)
    n_envs_list.sort(key=int)
    bar_labels.append(subprocess_type)
    colors_proxy[subprocess_type] = COLORS[idx]
    for n_envs in n_envs_list:
        csv_path = os.path.join(*[n_run_path, n_envs, 'progress.csv'])
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    fps_idx = row.index("time/fps")
                else:
                    fps_gen = int(row[fps_idx])
                line_count += 1
            
            if subprocess_type == "SubprocVecEnv":
                x_axis_label = "SubVE"
            else:
                x_axis_label = "DumVE"
            modes.append(f'{x_axis_label}_{n_envs}')
            
            bar_colors.append(COLORS[idx])
            
            fps.append(fps_gen)
            print(f'{subprocess_type} with {n_envs} fps_result:{fps_gen}')

# TODO colorizar por tipo, recortar nombre x para mejkor visualizacion

ind = np.arange(len(modes))
# plt.xticks(ind, modes)
# ax.set_xticks(ind)
# ax.set_xticklabels(modes)
# ax = fig.add_axes([0, 0, 1, 1])
print(modes)
print(bar_labels)
print(fps)
fig, ax = plt.subplots()
ax.set_ylabel('Vectorized Environments Types')
ax.set_ylabel('fps')

# ax.set_ylim(bottom=10)
# bar_labels = ['red', 'blue', '_red', 'orange']
ax.bar(modes, fps, color=bar_colors, label=bar_labels)



labels = list(colors_proxy.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=colors_proxy[label])
           for label in labels]
plt.legend(handles, labels)

plt.show()

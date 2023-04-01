import csv
import os

# TODO create matplotlib table

perf_folder = 'kinder-garten/perf'
subprocess_types = os.listdir(perf_folder)


for subprocess_type in subprocess_types:
    n_run_path = os.path.join(perf_folder, subprocess_type)
    n_envs_list = os.listdir(n_run_path)
    n_envs_list.sort(key=int)
    for n_envs in n_envs_list:
        csv_path = os.path.join(*[n_run_path, n_envs, 'progress.csv'])
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    fps_idx = row.index("time/fps")
                else:
                    print(f'{subprocess_type} with {n_envs} fps_result:{row[fps_idx]}')
                line_count += 1
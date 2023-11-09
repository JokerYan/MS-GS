import os
import re
import csv

from tensorboard.backend.event_processing import event_accumulator

# Path to the TensorBoard log directory
log_base_dir = '/home/zwyan/3d_cv/repos/gaussian-splatting/output'
scene_name_list = [
    "garden",
    "flowers", "treehill",
    "bicycle",
    "counter",
    "kitchen",
    "room",
    "stump", "bonsai",
]
exp_name_list = [
    'base',
    'ms',
    'abl_ms',
    'abl_fs',
    'abl_il',
]
exp_full_name_list = [
    "3D Gaussian\cite{kerbl3Dgaussians}",
    "Our Method",
    "3DGS + MS Train",
    "3DGS + Filter Small",
    "3DGS + Insert Large",
]
exp_idx_list = [0, 2, 3, 4, 1]
output_scale_list = [1, 4, 16, 64]
output_data_type_list = ['psnr', 'lpips', 'time']

def extract_scale(tag):
    scale_regex = re.compile(r'[/_]s(.*)\.0')
    scale = scale_regex.search(tag).group(1)
    return int(scale)

# this script is used to merge all results from the collect_results.py script to one csv file

# load all data
data_dict = {}
for scene_name in scene_name_list:
    scene_result_path = os.path.join(log_base_dir, scene_name, 'results.csv')
    # load
    with open(scene_result_path, 'r') as f:
        reader = csv.reader(f)
        scene_result = list(reader)

    # process first row
    titles = []
    for title in scene_result[0][1:]:
        scale = extract_scale(title)
        if 'psnr' in title:
            data_type = 'psnr'
        elif 'lpips' in title:
            data_type = 'lpips'
        elif 'time' in title:
            data_type = 'time'
        else:
            raise NotImplementedError
        titles.append((scale, data_type))

    for row in scene_result[1:]:
        exp_name = row[0]
        for column, data in enumerate(row[1:]):
            scale, data_type = titles[column]
            data_dict[(scene_name, exp_name, scale, data_type)] = data


# aggregate and calculate mean across scenes
output_rows = []
output_rows.append(['Scale'])
for scale in output_scale_list:
    output_rows[-1].append(f'{scale}x')
    for _ in range(len(output_data_type_list) - 1):
        output_rows[-1] .append('')
output_rows.append(['Metric'])
for scale in output_scale_list:
    for data_type in output_data_type_list:
        if data_type == 'psnr':
            title_str = r'PSNR$\uparrow$'
        elif data_type == 'lpips':
            title_str = r'LPIPS$\downarrow$'
        elif data_type == 'time':
            title_str = r'Time(ms)$\downarrow$'
        else:
            raise NotImplementedError
        output_rows[-1].append(r'\small{' + title_str + r'}')

for row_idx, exp_idx in enumerate(exp_idx_list):
    exp_name = exp_name_list[exp_idx]
    exp_full_name = exp_full_name_list[exp_idx]
    row_data = []
    for scale in output_scale_list:
        for data_type in output_data_type_list:
            data_list = []
            for scene_name in scene_name_list:
                data = data_dict[(scene_name, exp_name, scale, data_type)]
                data_list.append(float(data))
            data_mean = sum(data_list) / len(data_list)
            if data_type == 'psnr':
                data_str = f'{data_mean:.2f}'
            elif data_type == 'lpips':
                if data_mean == 0:
                    data_str = 'N.A.'
                else:
                    data_str = f'{data_mean:.3f}'
            elif data_type == 'time':
                data_str = f'{data_mean * 1000:.1f}'
            else:
                raise NotImplementedError
            # row_data.append(f'{data_type}_{scale}:{data_str}')
            row_data.append(data_str)
    output_rows.append([exp_full_name] + row_data)

# write csv
output_path = os.path.join(log_base_dir, 'results_all.csv')
with open(output_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(output_rows)
print(f'csv saved to {output_path}')
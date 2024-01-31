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

    # "train", "truck",
    # "drjohnson", "playroom",
]
exp_name_list = [
    # 'base',
    # 'ms',
    # 'abl_ms',
    # 'abl_fs',
    # 'abl_il',

    "base_interp_scale",
    "ms_only_interp_scale",
    "ms_interp_scale",
]
# result_file_name = 'results.csv'
result_file_name = 'results_interp_scale.csv'

for scene_name in scene_name_list:
    print('Scene:', scene_name)
    data_dict_list = []
    for exp_name in exp_name_list:
    # exp_name = 'ms'
        log_dir = os.path.join(log_base_dir, scene_name, exp_name)

        # Initialize an event accumulator
        ea = event_accumulator.EventAccumulator(log_dir,
            size_guidance={ # see below regarding this argument
                # event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                # event_accumulator.IMAGES: 0,
                # event_accumulator.AUDIO: 0,
                event_accumulator.SCALARS: 0,
                # event_accumulator.HISTOGRAMS: 0,
            })

        # Load all events from disk
        print('Loading events from:', log_dir)
        ea.Reload()
        print('Finished loading events.')

        # Get all tags from the log file
        tags = ea.Tags()['scalars']
        psnr_regex = re.compile(r'test.*psnr')
        lpips_regex = re.compile(r'test.*lpips')
        time_regex = re.compile(r'test.*time')

        matched_tags = []
        psnr_tag_list = []
        lpips_tag_list = []
        time_tag_list = []
        for tag in tags:
            if psnr_regex.search(tag):
                psnr_tag_list.append(tag)
            if lpips_regex.search(tag):
                lpips_tag_list.append(tag)
            if time_regex.search(tag):
                time_tag_list.append(tag)

        # sort using regex
        def extract_scale(tag):
            scale_regex = re.compile(r'/s(.*)\.0')
            scale = scale_regex.search(tag).group(1)
            return int(scale)
        psnr_tag_list.sort(key=extract_scale)
        lpips_tag_list.sort(key=extract_scale)
        time_tag_list.sort(key=extract_scale)
        matched_tags = psnr_tag_list + lpips_tag_list + time_tag_list

        # extract data
        data_dict = {}
        for tag in matched_tags:
            scalars = ea.Scalars(tag)
            data = scalars[-1].value
            data_dict[tag] = data
        data_dict_list.append(data_dict)

    print(data_dict_list)

    # save to csv
    csv_data = []
    first_row = ['exp_name'] + list(data_dict_list[0].keys())
    csv_data.append(first_row)
    for i in range(len(data_dict_list)):
        exp_name = exp_name_list[i]
        data_dict = data_dict_list[i]
        csv_row = [exp_name]
        for data in data_dict.values():
            csv_row.append(data)
        csv_data.append(csv_row)
    output_path = os.path.join(log_base_dir, scene_name, result_file_name)
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    print('Saved to:', output_path)

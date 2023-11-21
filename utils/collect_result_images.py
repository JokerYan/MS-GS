import io
import os
import re
import csv

from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tensorboard.backend.event_processing import event_accumulator

# Path to the TensorBoard log directory
log_base_dir = '/home/zwyan/3d_cv/repos/gaussian-splatting/output'
# scene_name_list = [
#     "garden",
#     "flowers", "treehill",
#     "bicycle",
#     "counter",
#     "kitchen",
#     "room",
#     "stump", "bonsai",
# ]
scene_name = 'bonsai'
exp_name_list = [
    # 'base',
    'ms',
    'abl_ms',
    'abl_fs',
    'abl_il',
]
target_scale_list = [
    1, 4, 8, 16, 32, 64, 128
]
display_time = False
# width_ratio = 0.6    # percentage of width for cropping each image
# width_ratio = 0.75    # percentage of width for cropping each image
width_ratio = 1.0    # percentage of width for cropping each image
rotate = False

def add_text_to_image(img, text):
    """
    Add text to the bottom right corner of an image object.

    :param img: PIL Image object.
    :param text: Text to add to the image.
    :return: PIL Image object with text added.
    """
    draw = ImageDraw.Draw(img)

    # Default settings
    width = img.width
    font_size = int(width / 6)
    right_margin = font_size // 3
    bottom_margin = font_size // 3
    text_color = "white"
    font_path = "arial.ttf"  # Ensure this font is available or provide a full path

    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        print("Font file not found. Using default font.")
        font = ImageFont.load_default()

    # Calculate the text size and position
    text_size = draw.textbbox((0, 0), text, font=font)[2:]
    text_x = img.width - text_size[0] - right_margin
    text_y = img.height - text_size[1] - bottom_margin

    # Draw the text
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    return img

image_exp_name_scale_dict = {}          # 3 level dict, exp_name -> image_name -> scale -> image
time_exp_scale_dict = {}                # 2 level dict, exp_name -> scale -> time
for exp_name in exp_name_list:
    log_dir = os.path.join(log_base_dir, scene_name, exp_name)

    # Initialize an event accumulator
    ea = event_accumulator.EventAccumulator(log_dir,
        size_guidance={ # see below regarding this argument
            # event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            # event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,
            # event_accumulator.HISTOGRAMS: 0,
        })

    # Load all events from disk
    print('Loading events from:', log_dir)
    ea.Reload()
    print('Finished loading events.')

    # Get all tags from the log file
    image_tags = ea.Tags()['images']
    scalar_tags = ea.Tags()['scalars']
    image_regex = re.compile(r'test.*/(render|ground_truth)$')
    time_regex = re.compile(r'test.*time')

    matched_image_tags = []
    for tag in image_tags:
        if image_regex.search(tag):
            matched_image_tags.append(tag)
    matched_scalar_tags = []
    for tag in scalar_tags:
        if time_regex.search(tag):
            matched_scalar_tags.append(tag)

    # sort using regex
    def extract_scale(tag):
        scale_regex = re.compile(r'[/_]s(.*)\.0')
        scale = scale_regex.search(tag).group(1)
        return int(scale)
    def extract_image_name(tag):
        name_regex = re.compile(r'test.*view_+(.*)/(.*)')
        name = name_regex.search(tag).group(1)
        render_name = name_regex.search(tag).group(2)
        is_gt = "ground_truth" in render_name
        return name, is_gt

    matched_image_tags = sorted(matched_image_tags, key=extract_scale)

    # extract image
    image_name_reso_dict = {}
    image_name_reso_dict_gt = {}
    for tag in matched_image_tags:
        image_name, is_gt = extract_image_name(tag)

        img_info = ea.Images(tag)[-1]
        img_bytes = img_info.encoded_image_string
        img_stream = io.BytesIO(img_bytes)
        image = Image.open(img_stream)

        if rotate:
            image = image.rotate(90, expand=True)

        scale = extract_scale(tag)
        if scale not in target_scale_list:
            continue
        if not is_gt:
            if image_name not in image_name_reso_dict:
                image_name_reso_dict[image_name] = {}
            # print(f'getting image from {tag}: {image_name} {scale}')
            image_name_reso_dict[image_name][scale] = image
        if is_gt:
            if image_name not in image_name_reso_dict_gt:
                image_name_reso_dict_gt[image_name] = {}
            image_name_reso_dict_gt[image_name][scale] = image

    if 'gt' not in image_exp_name_scale_dict:
        image_exp_name_scale_dict['gt'] = image_name_reso_dict_gt
    image_exp_name_scale_dict[exp_name] = image_name_reso_dict

    # extract time
    time_reso_dict = {}
    for tag in matched_scalar_tags:
        time = ea.Scalars(tag)[-1].value
        scale = extract_scale(tag)
        if scale not in target_scale_list:
            continue
        if scale not in time_reso_dict:
            time_reso_dict[scale] = time
    time_exp_scale_dict[exp_name] = time_reso_dict

# get the standard resolution
sample_image = next(iter(image_exp_name_scale_dict[exp_name_list[0]].values()))[1]      # get scale 1 image
sample_image_reso = sample_image.size
vis_size = (sample_image_reso[0] // 4, sample_image_reso[1] // 4)
cr_vis_size = (int(vis_size[0] * width_ratio), vis_size[1])       # crop size for vis
border_vis_size = int(vis_size[0] * (1 - width_ratio) // 2)

# choose and concat
root = tk.Tk()
root.title('Choose image')

image_idx = 0
image_count = len(image_exp_name_scale_dict[exp_name_list[0]].keys())
full_image = None
exp_name_list = ['gt'] + exp_name_list

def update_image():
    global image_idx
    global full_image
    print(f'getting image {image_idx}')
    image_name = list(image_exp_name_scale_dict[exp_name_list[0]].keys())[image_idx]
    image_list = []
    for exp_name in exp_name_list:
        exp_image_list = []
        for scale in sorted(image_exp_name_scale_dict[exp_name][image_name]):
            image = image_exp_name_scale_dict[exp_name][image_name][scale]
            image = image.resize(vis_size, Image.NEAREST)
            # crop for 50% in the center
            image = image.crop((border_vis_size, 0, border_vis_size + cr_vis_size[0], cr_vis_size[1]))

            # get respective time
            if exp_name != 'gt' and display_time:
                time = time_exp_scale_dict[exp_name][scale]
                time_str = f'{time * 1000:5.1f}ms'
            else:
                time_str = ''

            add_text_to_image(image, f'{time_str} {scale}x')
            exp_image_list.append(image)

        # concat horizontally
        gap_width = 10
        total_width = (cr_vis_size[0] + gap_width) * len(exp_image_list) - gap_width
        exp_image = Image.new('RGB', (total_width, cr_vis_size[1]), (255, 255, 255))
        for i, image in enumerate(exp_image_list):
            exp_image.paste(image, (i * (cr_vis_size[0] + gap_width), 0))
        image_list.append(exp_image)
     # concat vertically
    gap_height = 10
    total_height = (cr_vis_size[1] + gap_height) * len(image_list) - gap_height
    full_image = Image.new('RGB', (image_list[0].size[0], total_height), (255, 255, 255))
    for i, image in enumerate(image_list):
        full_image.paste(image, (0, i * (cr_vis_size[1] + gap_height)))

# show image using tk
update_image()
photo = ImageTk.PhotoImage(full_image)
label = tk.Label(root, image=photo)
label.image = photo
label.pack()

# key control
def key(event):
    global image_idx
    if event.char == 'q':
        root.destroy()
        return
    elif event.char == '1':
        image_idx = max(0, image_idx - 1)
    elif event.char == '3':
        image_idx = min(image_count - 1, image_idx + 1)
    elif event.char == '4':
        image_idx = max(0, image_idx - 10)
    elif event.char == '6':
        image_idx = min(image_count - 1, image_idx + 10)

    # update image
    update_image()
    photo = ImageTk.PhotoImage(full_image)
    label.configure(image=photo)
    label.image = photo

root.bind('<Key>', key)
root.mainloop()

# save image
output_name = f'{"_".join(exp_name_list)}{"_t" if display_time else ""}_{int(width_ratio * 100)}.png'
output_path = os.path.join(log_base_dir, scene_name, output_name)
full_image.save(output_path)
print(f'Save to {output_path}')

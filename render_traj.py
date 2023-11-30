#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import copy
import time

import subprocess
import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import cv2
from scipy.spatial.transform import Rotation

# max_reso_pow = 7
# max_reso_pow = 5
max_reso_pow = 1
train_reso_scales = [2**i for i in range(max_reso_pow + 1)]        # 1~128
# test_reso_scales = train_reso_scales + [(2**i + 2**(i+1)) / 2 for i in range(max_reso_pow)]     # 1~128, include half scales
test_reso_scales = train_reso_scales    # without half scales
test_reso_scales = sorted(test_reso_scales)
full_reso_scales = sorted(list(set(train_reso_scales + test_reso_scales)))

def render_trajectory(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                      anti_alias=False, data_name=None, trajectory_name='circle',
                      sync=True, frame_rate=30):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=full_reso_scales)
        gaussians.pre_cat_feature()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # prune gaussians far from center
        if trajectory_name == 'leave':
            gaussians.filter_center(scene.cameras_extent, min_y=0)

        # view = scene.getTestCameras()[0]
        view_idx = 0
        view = copy.deepcopy(scene.getTestCameras(scale=test_reso_scales[0])[view_idx])
        ori_image_height, ori_image_width = view.image_height, view.image_width
        gs_scale = 1.0          # size of scale compared to the original size
        fade_size = 1.0
        reso_idx = 0
        if anti_alias:
            filter_small = True
            filter_large = True
        else:
            filter_small = False
            filter_large = False

        if trajectory_name == 'circle':
            trajectory = generate_circle_trajectory(scene)
        elif trajectory_name == 'leave':
            trajectory = generate_leave_trajectory(scene)
        else:
            raise NotImplementedError

        rgb_frames = []
        depth_frames = []
        render_times = []

        for i, camera_pose in enumerate(trajectory):
            R, T = camera_pose
            view.R = R
            view.T = T

            if trajectory_name == 'circle':
                min_reso_scale = 1
                max_reso_scale = 64
                scale_period = 100
                reso_scale = (np.sin((i / scale_period - 0.25) * np.pi * 2) + 1) / 2
                reso_scale = reso_scale * (max_reso_scale - min_reso_scale) + min_reso_scale
                img_height, img_width = int(ori_image_height // reso_scale), int(ori_image_width // reso_scale)
                # image = torch.zeros((3, img_height, img_width)).to(view.original_image)
                # image = cv2.resize(view.original_image, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                # view.original_image = image
                view.image_width = img_width
                view.image_height = img_height

            view.cal_transform()
            torch.cuda.synchronize()
            time_start = time.perf_counter()

            results = render(view, gaussians, pipeline, background, scaling_modifier=gs_scale,
                             filter_small=filter_small, filter_large=filter_large, fade_size=fade_size)
            torch.cuda.synchronize()
            render_time = time.perf_counter() - time_start

            render_times.append(render_time)

            rendering = results["render"]
            acc_pixel_size = results["acc_pixel_size"]
            depth = results["depth"]

            rendering = torch.permute(rendering, (1, 2, 0))    # HWC
            rendering = rendering.cpu().numpy()
            rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)

            # normalize acc_pixel_size
            acc_pixel_size = torch.clip(acc_pixel_size / 10, 0, 1)
            acc_pixel_size = acc_pixel_size.cpu().numpy()

            # normalize depth
            depth = torch.clip(depth / torch.max(depth), 0, 1)
            depth = depth.cpu().numpy()
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

            rendering = cv2.resize(rendering, (ori_image_width, ori_image_height), interpolation=cv2.INTER_NEAREST)
            acc_pixel_size = cv2.resize(acc_pixel_size, (ori_image_width, ori_image_height), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (ori_image_width, ori_image_height), interpolation=cv2.INTER_NEAREST)

            if sync:
                text_right = True
                text_small = False
                text = f'{render_time * 1000:.2f}ms'
            else:
                method_full_name = 'Ours' if anti_alias else '3D-GS'
                text_small = True
                if anti_alias:
                    text_right = True
                    text = f'{render_time * 1000:.2f}ms {method_full_name}'
                else:
                    text_right = False
                    text = f'{method_full_name} {render_time * 1000:.2f}ms'

            rendering = add_text_to_image(rendering, text, right=text_right, small=text_small)
            depth = add_text_to_image(depth, text, right=text_right, small=text_small)

            cv2.imshow("acc_pixel_size", acc_pixel_size)
            cv2.imshow("depth", depth)
            cv2.imshow("rendering", rendering)
            cv2.setWindowTitle("rendering", f"{render_time * 1000:.2f}ms")
            cv2.waitKey(30)

            rgb_frames.append(rendering)
            depth_frames.append(depth)

    # convert video frames into time synced frames
    if sync:
        frame_count = len(rgb_frames)
        frame_interval = 1 / frame_rate
        cur_time = 0
        acc_render_time = 0
        idx = 0
        rgb_frames_sync = []
        depth_frames_sync = []
        while idx < frame_count:
            cur_time += frame_interval
            if cur_time >= acc_render_time:
                acc_render_time += render_times[idx]
                rgb_frames_sync.append(rgb_frames[idx])
                depth_frames_sync.append(depth_frames[idx])
                idx += 1
                continue
            rgb_frames_sync.append(rgb_frames[idx])
            depth_frames_sync.append(depth_frames[idx])
        rgb_frames = rgb_frames_sync
        depth_frames = depth_frames_sync

    # save video
    output_root = '/home/zwyan/3d_cv/papers/my papers/anti-aliasing/videos'
    output_dir = os.path.join(output_root, data_name)
    if not sync:
        output_dir = os.path.join(output_dir, 'async')
    makedirs(output_dir, exist_ok=True)
    method_name = 'ms' if anti_alias else 'base'
    rgb_output_name = f'{data_name}_{method_name}_{trajectory_name}_rgb.mp4'
    rgb_output_path = os.path.join(output_dir, rgb_output_name)
    rgb_height, rgb_width, _ = rgb_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    rgb_video = cv2.VideoWriter(rgb_output_path, fourcc, frame_rate, (rgb_width, rgb_height))
    for frame in tqdm(rgb_frames):
        # convert frame to uint8
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        rgb_video.write(frame)
    rgb_video.release()

    depth_output_name = f'{data_name}_{method_name}_{trajectory_name}_depth.mp4'
    depth_output_path = os.path.join(output_dir, depth_output_name)
    depth_height, depth_width, _ = depth_frames[0].shape
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    depth_video = cv2.VideoWriter(depth_output_path, fourcc, frame_rate, (depth_width, depth_height))
    for frame in tqdm(depth_frames):
        # convert frame to uint8
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        depth_video.write(frame)
    depth_video.release()

    print(rgb_frames[0].shape, depth_frames[0].shape)

def generate_circle_trajectory(scene):
    view_idx = 1
    view = copy.deepcopy(scene.getTestCameras(scale=test_reso_scales[0])[view_idx])
    reference_position = view.T.squeeze()

    num_steps = 300
    angle_step = 2 * np.pi / 100
    trajectory = []
    for step in range(num_steps):
        angle = step * angle_step

        radius = np.linalg.norm(reference_position)

        dx = radius * np.cos(angle)
        dz = radius * np.sin(angle)
        dy = radius * 0.1 * np.cos(angle + np.pi) * 0
        C = np.array([dx, dy, dz])

        # look_at = np.array([0, 0, 0])
        look_at = np.array([0, 1, 0])
        rotation_matrix = pos_to_rotation(C, look_at)
        # rotation_matrix = np.eye(3)

        rotation_matrix = rotation_matrix.T
        translation = - rotation_matrix @ C

        trajectory.append((rotation_matrix, translation))
    return trajectory


def generate_leave_trajectory(scene):
    view_idx = 1
    view = copy.deepcopy(scene.getTestCameras(scale=test_reso_scales[0])[view_idx])

    num_steps = 150
    trajectory = []
    R, T = view.R, view.T
    for step in range(num_steps):
        T = T * np.array([0, 0, 1.05])
        trajectory.append((R, T.copy()))
    return trajectory


def add_text_to_image(image, text, bottom=True, right=True, small=False):
    height, width, channel = image.shape
    image = np.copy(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if small:
        font_scale = 2
        thickness = 2
    else:
        font_scale = 3
        thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    text_pos_height = height - text_height if bottom else 30
    text_pos_width = width - text_width - 30 if right else 10
    text_pos = (text_pos_width, text_pos_height)

    image = cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return image

def pos_to_rotation(position_vector, look_at):
    # Z-axis (camera is pointing in the positive Z direction)
    z_axis = -(position_vector - look_at).copy()
    # z_axis = position_vector.copy()
    z_axis /= np.linalg.norm(z_axis)  # Normalize
    # Y-axis (up direction)
    y_axis = np.array([0, -1, 0])
    # X-axis (right direction)
    x_axis = np.cross(z_axis, y_axis)
    x_axis /= np.linalg.norm(x_axis)
    # Recompute the Y-axis to ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)

    # Assemble the rotation matrix
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # rotation_matrix = rotation_matrix.transpose()

    return rotation_matrix

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--anti_alias", action="store_true", default=False)
    parser.add_argument("--no_sync", action="store_true", default=False)
    parser.add_argument('--frame_rate', type=int, default=30)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    data_name = os.path.basename(args.source_path)
    trajectory_name = 'circle'
    # trajectory_name = 'leave'
    sync = not args.no_sync

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    render_trajectory(model.extract(args), args.iteration, pipeline.extract(args),
                      anti_alias=args.anti_alias, data_name=data_name, trajectory_name=trajectory_name,
                      sync=sync, frame_rate=args.frame_rate)
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

def render_trajectory(dataset: ModelParams, iteration: int, pipeline: PipelineParams, anti_alias=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=full_reso_scales)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # # prune gaussians far from center
        # gaussians.filter_center(scene.cameras_extent)

        # view = scene.getTestCameras()[0]
        view_idx = 0
        view = copy.deepcopy(scene.getTestCameras(scale=test_reso_scales[0])[view_idx])
        gs_scale = 1.0          # size of scale compared to the original size
        fade_size = 1.0
        reso_idx = 0
        view_resolution = None
        if anti_alias:
            filter_small = True
            filter_large = True
        else:
            filter_small = False
            filter_large = False

        trajectory = generate_circle_trajectory(scene)

        for i, camera_pose in enumerate(trajectory):
            R, T = camera_pose
            view.R = R
            view.T = T

            view.cal_transform()

            rotation = Rotation.from_matrix(R)
            euler = rotation.as_euler('xyz', degrees=True)
            print(i, euler, view.T)

            torch.cuda.synchronize()
            time_start = time.time()

            results = render(view, gaussians, pipeline, background, scaling_modifier=gs_scale,
                             filter_small=filter_small, filter_large=filter_large, fade_size=fade_size)
            rendering = results["render"]
            acc_pixel_size = results["acc_pixel_size"]
            depth = results["depth"]

            torch.cuda.synchronize()
            render_time = time.time() - time_start

            rendering = torch.permute(rendering, (1, 2, 0))    # HWC
            rendering = rendering.cpu().numpy()
            rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)

            # normalize acc_pixel_size
            acc_pixel_size = torch.clip(acc_pixel_size / 10, 0, 1)
            acc_pixel_size = acc_pixel_size.cpu().numpy()

            # normalize depth
            depth = torch.clip(depth / torch.max(depth), 0, 1)
            depth = depth.cpu().numpy()

            if view_resolution is None:
                view_resolution = rendering.shape[:2]
            else:
                rendering = cv2.resize(rendering, view_resolution[::-1], interpolation=cv2.INTER_NEAREST)
                acc_pixel_size = cv2.resize(acc_pixel_size, view_resolution[::-1], interpolation=cv2.INTER_NEAREST)
                depth = cv2.resize(depth, view_resolution[::-1], interpolation=cv2.INTER_NEAREST)

            cv2.imshow("acc_pixel_size", acc_pixel_size)
            cv2.imshow("depth", depth)
            cv2.imshow("rendering", rendering)
            cv2.setWindowTitle("rendering", f"{render_time * 1000:.2f}ms")
            cv2.waitKey(0)


def generate_circle_trajectory(scene):
    view_idx = 1
    view = copy.deepcopy(scene.getTestCameras(scale=test_reso_scales[0])[view_idx])
    reference_position = view.T.squeeze()

    num_steps = 200
    angle_step = 4 * np.pi / num_steps
    trajectory = []
    for step in range(num_steps):
        angle = step * angle_step

        # # Calculate the new position with rotation around the Y-axis
        # rotation_y = np.array([
        #     [np.cos(angle),  0, np.sin(angle)],
        #     [0,              1, 0],
        #     [-np.sin(angle), 0, np.cos(angle)]
        # ])
        # position_vector = rotation_y @ reference_position

        radius = np.linalg.norm(reference_position)

        dx = radius * np.cos(angle)
        dz = radius * np.sin(angle)
        dy = radius * 0.1 * np.cos(angle + np.pi) * 0
        C = np.array([dx, dy, dz])

        look_at = np.array([0, 1, 0])
        rotation_matrix = pos_to_rotation(C, look_at)
        # rotation_matrix = np.eye(3)

        rotation_matrix = rotation_matrix.T
        translation = - rotation_matrix @ C

        trajectory.append((rotation_matrix, translation))
    return trajectory


# def pos_to_rotation(position_vector):
#     # Z-axis (camera is pointing in the positive Z direction)
#     # z_axis = -position_vector.copy()
#     z_axis = position_vector.copy()
#     z_axis /= np.linalg.norm(z_axis)  # Normalize
#     # Y-axis (up direction)
#     y_axis = np.array([0, 1, 0])
#     # X-axis (right direction)
#     x_axis = np.cross(y_axis, z_axis)
#     x_axis /= np.linalg.norm(x_axis)
#     # Recompute the Y-axis to ensure orthogonality
#     y_axis = np.cross(z_axis, x_axis)
#
#     # Assemble the rotation matrix
#     rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
#
#     # rotation_matrix = rotation_matrix.transpose()
#
#     return rotation_matrix

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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    render_trajectory(model.extract(args), args.iteration, pipeline.extract(args), anti_alias=args.anti_alias)
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

max_reso_pow = 7
# max_reso_pow = 5
# max_reso_pow = 1
train_reso_scales = [2**i for i in range(max_reso_pow + 1)]        # 1~128
# test_reso_scales = train_reso_scales + [(2**i + 2**(i+1)) / 2 for i in range(max_reso_pow)]     # 1~128, include half scales
test_reso_scales = train_reso_scales    # without half scales
test_reso_scales = sorted(test_reso_scales)
full_reso_scales = sorted(list(set(train_reso_scales + test_reso_scales)))

def render_interactive(dataset: ModelParams, iteration: int, pipeline: PipelineParams, anti_alias=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=full_reso_scales)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # prune gaussians far from center
        gaussians.filter_center(5)

        # view = scene.getTestCameras()[0]
        view_idx = 0
        view = copy.deepcopy(scene.getTestCameras(scale=test_reso_scales[0])[view_idx])
        gs_scale = 1.0          # size of scale compared to the original size
        fade_size = 1.0
        reso_idx = 0
        view_resolution = None
        if anti_alias:
            filter_small = True
        else:
            filter_small = False
        while True:
            view.cal_transform()
            torch.cuda.synchronize()
            time_start = time.time()

            results = render(view, gaussians, pipeline, background, scaling_modifier=gs_scale,
                             filter_small=filter_small, fade_size=fade_size)
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
                rendering = cv2.resize(rendering, view_resolution[::-1])
                acc_pixel_size = cv2.resize(acc_pixel_size, view_resolution[::-1])
                depth = cv2.resize(depth, view_resolution[::-1])

            cv2.imshow("rendering", rendering)
            cv2.setWindowTitle("rendering", f"{render_time * 1000:.2f}ms")
            cv2.imshow("acc_pixel_size", acc_pixel_size)
            cv2.imshow("depth", depth)

            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('4'):
                view.T[0] += 0.1
            elif key == ord('6'):
                view.T[0] -= 0.1
            elif key == ord('8'):
                view.T[1] += 0.1
            elif key == ord('2'):
                view.T[1] -= 0.1
            elif key == ord('7'):
                view.T[2] += 0.5
            elif key == ord('9'):
                view.T[2] -= 0.5
            elif key == ord('1'):
                view.scale *= 0.9
            elif key == ord('3'):
                view.scale /= 0.9
            elif key == ord('['):
                gs_scale = max(0.1, gs_scale - 0.1)
            elif key == ord(']'):
                gs_scale = min(2.0, gs_scale + 0.1)
            elif key == ord(';'):
                fade_size = max(0.1, fade_size - 0.1)
            elif key == ord('\''):
                fade_size = min(2.0, fade_size + 0.1)
            elif key == ord('x'):
                view_idx = (view_idx - 1) % len(scene.getTestCameras())
                view = copy.deepcopy(scene.getTestCameras(scale=test_reso_scales[reso_idx])[view_idx])
            elif key == ord('c'):
                view_idx = (view_idx + 1) % len(scene.getTestCameras())
                view = copy.deepcopy(scene.getTestCameras(scale=test_reso_scales[reso_idx])[view_idx])
            elif key == ord('z'):
                view = copy.deepcopy(scene.getTestCameras(scale=test_reso_scales[reso_idx])[view_idx])
                gs_scale = 1.0
                fade_size = 1.0
                reso_idx = 0
            elif key == ord('/'):
                reso_idx = min(reso_idx + 1, max_reso_pow)
                view = copy.deepcopy(scene.getTestCameras(scale=test_reso_scales[reso_idx])[view_idx])
            elif key == ord('.'):
                reso_idx = max(reso_idx - 1, 0)
                view = copy.deepcopy(scene.getTestCameras(scale=test_reso_scales[reso_idx])[view_idx])

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
    render_interactive(model.extract(args), args.iteration, pipeline.extract(args), anti_alias=args.anti_alias)
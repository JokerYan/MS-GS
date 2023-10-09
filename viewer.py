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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


def render_interactive(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # view = scene.getTestCameras()[0]
        view = copy.deepcopy(scene.getTrainCameras()[0])
        gs_scale = 1.0          # size of scale compared to the original size
        while True:
            view.cal_transform()
            torch.cuda.synchronize()
            time_start = time.time()

            rendering = render(view, gaussians, pipeline, background, scaling_modifier=gs_scale)["render"]

            torch.cuda.synchronize()
            render_time = time.time() - time_start

            rendering = torch.permute(rendering, (1, 2, 0))    # HWC
            rendering = rendering.cpu().numpy()
            rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
            cv2.imshow("rendering", rendering)
            cv2.setWindowTitle("rendering", f"{render_time * 1000:.2f}ms")
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
            elif key == ord('z'):
                view = copy.deepcopy(scene.getTrainCameras()[0])
                gs_scale = 1.0

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    render_interactive(model.extract(args), args.iteration, pipeline.extract(args))
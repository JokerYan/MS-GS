import os
import sys
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import torch
from gaussian_renderer import render, network_gui

from train import training

mipnerf_scene_list = ["garden", "flowers", "treehill", "bicycle", "counter", "kitchen", "room", "stump", "bonsai"]
tnt_scene_list = ["truck", "train"]

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--test_interval", type=int, default=5_000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    scene_list = [
        "garden",
        "flowers", "treehill",
        "bicycle",
        "counter",
        "kitchen",
        "room",
        "stump", "bonsai",
    ]
    method_dict = {
        'ms': {         # our ms model
            "ms_train": True,
            "filter_small": True,
            "prune_small": False,
            "grow_large": False,
            "multi_occ": False,
            "multi_dc": False,
            "preserve_large": False,
            "insert_large": True,
            "iterations": 40000,
            "densify_until_iter": 15000,
        },
        "base": {
            "ms_train": False,
            "filter_small": False,
            "prune_small": False,
            "grow_large": False,
            "multi_occ": False,
            "multi_dc": False,
            "preserve_large": False,
            "insert_large": False,
            "iterations": 30000,
            "densify_until_iter": 15000,
        },
        # 'abl_ms': {         # ablation for using ms train only
        #     "ms_train": True,
        #     "filter_small": False,
        #     "prune_small": False,
        #     "grow_large": False,
        #     "multi_occ": False,
        #     "multi_dc": False,
        #     "preserve_large": False,
        #     "insert_large": False,
        #     "iterations": 40000,
        #     "densify_until_iter": 15000,
        # },
        # 'abl_fs': {         # ablation for ms_train and filter small
        #     "ms_train": True,
        #     "filter_small": True,
        #     "prune_small": False,
        #     "grow_large": False,
        #     "multi_occ": False,
        #     "multi_dc": False,
        #     "preserve_large": False,
        #     "insert_large": False,
        #     "iterations": 40000,
        #     "densify_until_iter": 15000,
        # },
        # 'abl_il': {         # ablation for ms_train and insert large
        #     "ms_train": True,
        #     "filter_small": False,
        #     "prune_small": False,
        #     "grow_large": False,
        #     "multi_occ": False,
        #     "multi_dc": False,
        #     "preserve_large": False,
        #     "insert_large": True,
        #     "iterations": 40000,
        #     "densify_until_iter": 15000,
        # },
    }

    source_dir = args.source_path
    model_dir = args.model_path

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    for scene in scene_list:
        for method in method_dict:
            args.ms_train = method_dict[method]["ms_train"]
            args.filter_small = method_dict[method]["filter_small"]
            args.prune_small = method_dict[method]["prune_small"]
            args.preserve_large = method_dict[method]["grow_large"]
            args.multi_occ = method_dict[method]["multi_occ"]
            args.multi_dc = method_dict[method]["multi_dc"]
            args.preserve_large = method_dict[method]["preserve_large"]
            args.insert_large = method_dict[method]["insert_large"]
            args.iterations = method_dict[method]["iterations"]
            args.densify_until_iter = method_dict[method]["densify_until_iter"]
            if args.iterations not in args.save_iterations:
                args.save_iterations.append(args.iterations)

            args.source_path = os.path.join(source_dir, scene)
            args.model_path = os.path.join(model_dir, scene, method)

            torch.cuda.empty_cache()    # Free up memory before training, hopefully this will work
            training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.test_interval,
                     args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
                     ms_train=args.ms_train, filter_small=args.filter_small, prune_small=args.prune_small,
                     insert_large=args.insert_large,
                     preserve_large=args.preserve_large, multi_occ=args.multi_occ, multi_dc=args.multi_dc)

    # source_path
    # model_path

    # All done
    print("\nTraining complete.")
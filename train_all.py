import sys
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import torch
from gaussian_renderer import render, network_gui

from train import training

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
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    scene_list = [
        # "bicycle",
        "bonsai", "counter", "garden", "kitchen", "room", "stump"
    ]
    method_dict = {
        'ms': {
            "ms_train": True,
            "prune_small": True,
            "grow_large": True,
        },
        "base": {

        }
    }

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.test_interval,
    #          args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
    #          ms_train=args.ms_train, filter_small=args.filter_small, prune_small=args.prune_small,
    #          preserve_large=args.preserve_large, multi_occ=args.multi_occ, multi_dc=args.multi_dc)

    # source_path
    # model_path

    # All done
    print("\nTraining complete.")
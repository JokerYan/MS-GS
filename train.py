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

import os
import time
import math

import cv2
import torch
from random import randint, random
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


max_reso_pow = 7
# max_reso_pow = 5
# max_reso_pow = 1
train_reso_scales = [2**i for i in range(max_reso_pow + 1)]        # 1~128
# test_reso_scales = train_reso_scales + [(2**i + 2**(i+1)) / 2 for i in range(max_reso_pow)]     # 1~128, include half scales
test_reso_scales = train_reso_scales    # without half scales
test_reso_scales = sorted(test_reso_scales)
full_reso_scales = sorted(list(set(train_reso_scales + test_reso_scales)))
print('train_reso_scales', train_reso_scales)
print('test_reso_scales', test_reso_scales)
print('full_reso_scales', full_reso_scales)

ms_from_iter = 1
# ms_from_iter = 15000

def training(
        dataset, opt, pipe, testing_iterations, test_interval,
        saving_iterations, checkpoint_iterations, checkpoint, debug_from,
        ms_train=False, filter_small=False, prune_small=False, preserve_large=False,
        multi_occ=False, multi_dc=False,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, reso_lvls=len(train_reso_scales), multi_occ=multi_occ, multi_dc=multi_dc)
    scene = Scene(dataset, gaussians, resolution_scales=full_reso_scales)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    reso_idx = 0
    reso_iterations = [0 for _ in range(len(train_reso_scales))]
    for iteration in range(first_iter, opt.iterations + 1):
        # if iteration < opt.densify_until_iter + 10000:
        #     fade_size = 0
        # else:
        #     fade_size = 1.0
        fade_size = 0

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer,
                                       filter_small=filter_small, fade_size=fade_size)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            if ms_train and iteration >= ms_from_iter:
            # if ms_train and iteration > opt.densify_until_iter:
            # if ms_train and iteration > 5000:
            # if ms_train:
            #     resolution_scale = train_reso_scales[randint(0, len(train_reso_scales)-1)]

                # half the chance of getting the highest resolution
                if random() < 0.5:
                    reso_idx = 0
                else:
                    reso_idx = randint(0, len(train_reso_scales)-1)
            else:
                reso_idx = 0  # use the highest resolution only
            resolution_scale = train_reso_scales[reso_idx]
            viewpoint_stack = scene.getTrainCameras(resolution_scale).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        reso_iterations[reso_idx] += 1

        # if iteration == opt.densify_until_iter:
        if iteration == ms_from_iter:
            gaussians.start_ms_lr()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, filter_small=filter_small, fade_size=fade_size)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        pixel_sizes = render_pkg["pixel_sizes"]

        # if iteration % 500 == 0:
        #     # print(iteration, torch.mean(gaussians.get_occ_multiplier),
        #     #       torch.min(gaussians.get_occ_multiplier), torch.max(gaussians.get_occ_multiplier))
        #     print(iteration, torch.mean(gaussians._occ_multiplier),
        #           torch.min(gaussians._occ_multiplier), torch.max(gaussians._occ_multiplier))
        #     print(iteration, torch.mean(gaussians._dc_delta),
        #           torch.min(gaussians._dc_delta), torch.max(gaussians._dc_delta))
        #     # print()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if ms_train:
            loss_multiplier = 1 / (reso_idx + 1)
            loss *= loss_multiplier
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, test_interval, scene, render, (pipe, background),
                            filter_small, fade_size)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # if iteration > opt.densify_from_iter:
            if preserve_large and iteration > opt.densify_until_iter:
                if resolution_scale == train_reso_scales[-1]:
                    gaussians.update_base_gaussian_mask(visibility_filter)

            # if iteration % 100 == 0:
            #     print(torch.min(gaussians.min_pixel_sizes), torch.max(gaussians.min_pixel_sizes),
            #           torch.median(gaussians.min_pixel_sizes), torch.min(pixel_sizes), torch.max(pixel_sizes))

            # Densification
            gaussians.update_pixel_sizes(visibility_filter, pixel_sizes, reso_idx, iteration)

            if reso_idx == 0:
                assert True
            elif reso_idx == 4:
                v_mask = torch.logical_and(visibility_filter, gaussians.target_reso_lvl == 0)
                p_mask = torch.logical_and(pixel_sizes > 0, gaussians.target_reso_lvl == 0)
                if iteration % 100 == 0:
                    print("filter ratio:", torch.mean(v_mask.float()) / torch.mean(p_mask.float()))
                assert True

            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, reso_lvl=reso_idx)

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if iteration > opt.densify_from_iter and reso_iterations[reso_idx] % opt.densification_interval == 0:
                        # and resolution_scale == train_reso_scales[0]:       # densify only at the highest resolution
                    if reso_idx == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    else:
                        gaussians.grow_large_gaussians(opt.densify_grad_threshold, reso_idx)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                if prune_small and iteration > opt.densify_from_iter and iteration % 1000 == 0:
                    gaussians.prune_small_points()

            # # Add large gaussians
            # if iteration < 15000:
            #     if iteration > 500 and iteration % 100 == 0:
            #         added_new = gaussians.add_large_gaussian(viewpoint_cam, render_pkg)
            #         #
            #         # if added_new and iteration >= 1000:
            #         #     last_image = image.clone()
            #         #     old_pixel_size = render_pkg["acc_pixel_size"] / 3.0
            #         #     old_pixel_size = torch.tile(old_pixel_size, (3, 1, 1))
            #         #     old_depth = render_pkg["depth"]
            #         #     render_pkg = render(viewpoint_cam, gaussians, pipe, background, fade_size=fade_size)
            #         #     new_image = render_pkg["render"]
            #         #     new_depth = render_pkg["depth"]
            #         #     max_depth = torch.maximum(torch.max(old_depth), torch.max(new_depth))
            #         #     old_depth = old_depth / max_depth
            #         #     new_depth = new_depth / max_depth
            #         #     old_depth = torch.tile(old_depth, (3, 1, 1))
            #         #     new_depth = torch.tile(new_depth, (3, 1, 1))
            #         #
            #         #     image = torch.cat((last_image, new_image, old_pixel_size, old_depth, new_depth), dim=-1)
            #         #     image = image.cpu().numpy()
            #         #     image = image.transpose(1, 2, 0)
            #         #     image = cv2.resize(image, (image.shape[1] * resolution_scale // 2, image.shape[0] * resolution_scale // 2), interpolation=cv2.INTER_NEAREST)
            #         #     cv2.imshow("image", image)
            #         #     cv2.waitKey(0)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, test_interval, scene : Scene, renderFunc, renderArgs,
                    filter_small=False, fade_size=0):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations or (iteration > 0 and iteration % test_interval == 0):
        torch.cuda.empty_cache()
        validation_configs = []
        for reso_scale in test_reso_scales:
            validation_configs.append({
                'name': 'test', 'cameras': scene.getTestCameras(reso_scale), 'scale': reso_scale
            })
        for reso_scale in train_reso_scales:
            validation_configs.append({
                'name': 'train', 'cameras': [scene.getTrainCameras(reso_scale)[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
                'scale': reso_scale
            })
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras(reso_scale)},
        #                       {'name': 'train', 'cameras' : [scene.getTrainCameras(reso_scale)[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        output_str = f"[ITER {iteration}] Evaluating:"
        for config in validation_configs:
            reso_scale = config['scale']
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                render_time = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    torch.cuda.synchronize()
                    start_time = time.time()

                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs,
                                                   filter_small=filter_small, fade_size=fade_size)["render"], 0.0, 1.0)

                    torch.cuda.synchronize()
                    render_time += time.time() - start_time

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            f"{config['name']}_s{reso_scale:.1f}_view_{viewpoint.image_name}/render",
                            image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f"{config['name']}_s{reso_scale:.1f}_view_{viewpoint.image_name}/ground_truth",
                                gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                render_time /= len(config['cameras'])
                # print(f"\n[ITER {iteration}] Evaluating {config['name']} s{reso_scale:.1f}: L1 {l1_test} PSNR {psnr_test}")
                output_str += f"s{reso_scale:.1f} PSNR {psnr_test:.2f} | "
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/s{reso_scale:.1f}_loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/s{reso_scale:.1f}_loss_viewpoint - psnr", psnr_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/s{reso_scale:.1f}_loss_viewpoint - render_time", render_time, iteration)
        print(output_str)

    if iteration % 1000 == 0:
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram("scene/pixel_sizes_histogram", torch.clip(scene.gaussians.max_pixel_sizes, max=10), iteration)
            for i in range(scene.gaussians.get_occ_multiplier.shape[1]):
                tb_writer.add_histogram(f"scene/occ_multiplier_histogram_{i}", scene.gaussians.get_occ_multiplier[:, i], iteration)
        torch.cuda.empty_cache()

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
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--ms_train', action='store_true', default=False, help='use multi-scale training')
    parser.add_argument('--filter_small', action='store_true', default=False, help='filter small gaussians based on pixel size')
    parser.add_argument('--prune_small', action='store_true', default=False, help='prune small gaussians based on pixel size')
    parser.add_argument('--preserve_large', action='store_true', default=False, help='preserve large gaussians')
    parser.add_argument('--multi_occ', action='store_true', default=False, help='use multiple occ multiplier')
    parser.add_argument('--multi_dc', action='store_true', default=False, help='use multiple dc features delta')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.test_interval,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
             ms_train=args.ms_train, filter_small=args.filter_small, prune_small=args.prune_small,
             preserve_large=args.preserve_large, multi_occ=args.multi_occ, multi_dc=args.multi_dc)

    # All done
    print("\nTraining complete.")

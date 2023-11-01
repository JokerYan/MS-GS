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
from random import randint, random, choice

from lpipsPyTorch import lpips
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
# ms_from_iter = 5000
# ms_from_iter = 15000

def training(
        dataset, opt, pipe, testing_iterations, test_interval,
        saving_iterations, checkpoint_iterations, checkpoint, debug_from,
        ms_train=False, filter_small=False, prune_small=False, preserve_large=False,
        multi_occ=False, multi_dc=False, grow_large=False, insert_large=False
):
    first_iter = 0
    last_reset_opacity_iter = None
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

    if insert_large:
        # increase reso at these iterations, starting from 1 (2x)
        # inc_reso_at = torch.tensor([5000 + 1000 * (i + 1) for i in range(len(train_reso_scales)-1)])
        # inc_reso_at = torch.tensor([1000 * (i + 1) for i in range(len(train_reso_scales)-1)])
        # inc_reso_idx = torch.tensor([(i + 1) for i in range(len(train_reso_scales)-1)])
        # inc_reso_at = torch.tensor([990])
        # inc_reso_idx = torch.tensor([4])
        # inc_reso_at = torch.tensor([940, 950, 960, 970, 980, 990])
        # inc_reso_idx = torch.tensor([2, 3, 4, 5, 6, 7])
        # inc_reso_at = torch.tensor([5000 - 20, 5000 - 30, 5000 - 10])
        base_iter = 1000
        # inc_reso_at = torch.tensor([base_iter - 30, base_iter - 20, base_iter - 10])
        inc_reso_at = torch.tensor([base_iter + 10, base_iter + 20, base_iter + 30])
        inc_reso_idx = torch.tensor([2, 4, 6])
        # inc_reso_idx_train = [[1, 2], [3, 4], [5, 6, 7]]
        inc_reso_idx_train = [[2, 3], [4, 5], [6, 7]]

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    reso_idx = 0
    reso_iterations = [0 for _ in range(len(train_reso_scales))]            # the number of iterations trained on each reso
    for iteration in range(first_iter, opt.iterations + 1):
        # if iteration < opt.densify_until_iter + 10000:
        #     fade_size = 0
        # else:
        #     fade_size = 1.0
        fade_size = 0
        filter_large = (grow_large or insert_large)

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer,
                                       filter_small=filter_small, filter_large=filter_large, fade_size=fade_size)["render"]
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
                if random() < 0.75:
                    reso_idx = 0
                else:
                    if insert_large:
                        # reso_idx_counter = 0
                        # for iter in inc_reso_at:
                        #     if iteration > iter:
                        #         reso_idx_counter += 1
                        # max_cur_reso_idx = 0 if reso_idx_counter == 0 else inc_reso_idx[reso_idx_counter-1]
                        # reso_idx = randint(0, max_cur_reso_idx)

                        reso_idx_mask = iteration > inc_reso_at
                        reso_idx_mask_list = torch.arange(0, len(inc_reso_at))[reso_idx_mask]
                        reso_idx_list = [0]
                        for idx in reso_idx_mask_list:
                            reso_idx_list += inc_reso_idx_train[idx]
                        # reso_idx_list = inc_reso_idx[reso_idx_mask].tolist() + [0]
                        # reso_idx = choice(reso_idx_list)

                        # choose the one of the least trained resolution
                        reso_iterations_list = [reso_iterations[idx] for idx in reso_idx_list]
                        min_reso_iterations = min(reso_iterations_list)
                        reso_idx_list = [reso_idx_list[i] for i in range(len(reso_idx_list)) if reso_iterations_list[i] == min_reso_iterations]
                        reso_idx = choice(reso_idx_list)

                        if reso_idx_mask.sum() > 0:
                            print("choosing reso idx:", reso_idx_list, reso_idx)
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
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, filter_small=filter_small,
                            filter_large=filter_large, fade_size=fade_size)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        pixel_sizes = render_pkg["pixel_sizes"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if ms_train:
            loss_multiplier = 1 / (reso_idx * 2 + 1)
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
                            filter_small, filter_large, fade_size)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # if iteration > opt.densify_from_iter:
            if preserve_large and iteration > opt.densify_until_iter:
                if resolution_scale == train_reso_scales[-1]:
                    gaussians.update_base_gaussian_mask(visibility_filter)

            # Densification
            if iteration >= 250 and (last_reset_opacity_iter is None or iteration - last_reset_opacity_iter > 250):
                gaussians.update_pixel_sizes(visibility_filter, pixel_sizes, reso_idx, iteration)

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
                        if grow_large:
                            gaussians.grow_large_gaussians(opt.densify_grad_threshold, reso_idx)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    last_reset_opacity_iter = iteration
                    gaussians.reset_opacity()

                if prune_small and iteration > opt.densify_from_iter and iteration % 1000 == 0:
                    gaussians.prune_small_points()

            if insert_large and iteration in inc_reso_at:
                torch.cuda.synchronize()
                insert_time = time.time()
                # base_reso_idx = inc_reso_at.index(iteration)
                base_reso_idx = 0       # always create from the highest resolution
                # base_reso_idx = max(inc_reso_at.index(iteration) - 2, 0)        # 3 lvls lower than the target reso
                # next_reso_idx = inc_reso_at.index(iteration) + 1
                # next_reso_idx = 4
                next_reso_idx = inc_reso_idx[inc_reso_at.tolist().index(iteration)]
                base_reso_cams = scene.getTrainCameras(train_reso_scales[base_reso_idx]).copy()
                next_reso_cams = scene.getTrainCameras(train_reso_scales[next_reso_idx]).copy()
                base_vis_filter_list = []
                next_vis_filter_list = []
                for cam in base_reso_cams:
                    render_out = render(cam, gaussians, pipe, background, filter_small=filter_small,
                                        filter_large=filter_large, fade_size=fade_size)
                    vis_filter = render_out["visibility_filter"]
                    base_vis_filter_list.append(vis_filter)

                pixel_size_threshold = 1
                min_pixel_sizes = torch.ones_like(gaussians.min_pixel_sizes) * pixel_size_threshold
                for i, cam in enumerate(next_reso_cams):
                    render_out = render(cam, gaussians, pipe, background, filter_small=filter_small,
                                        filter_large=filter_large, fade_size=fade_size)
                    vis_filter = render_out["visibility_filter"]
                    next_vis_filter_list.append(vis_filter)
                    min_pixel_sizes = torch.where(
                        torch.logical_and(render_out["pixel_sizes"] > 0, base_vis_filter_list[i]),
                        torch.minimum(render_out["pixel_sizes"], min_pixel_sizes),
                        min_pixel_sizes
                    )

                # compare the visibility filter of the same camera at each resolution
                all_diff_vis_filter = torch.zeros_like(base_vis_filter_list[0])
                ratios_sum = torch.tensor(0, dtype=torch.float32, device=all_diff_vis_filter.device)
                for i in range(len(base_vis_filter_list)):
                    diff_vis_filter = torch.logical_and(base_vis_filter_list[i], torch.logical_not(next_vis_filter_list[i]))
                    all_diff_vis_filter = torch.logical_or(all_diff_vis_filter, diff_vis_filter)
                    ratios_sum += torch.mean(diff_vis_filter.float())
                # print(f"avg ratio: {ratios_sum/len(base_vis_filter_list):.3f} of {len(base_vis_filter_list)}, total ratio: {torch.mean(all_diff_vis_filter.float()):.3f}")

                # all_diff_vis_filter = min_pixel_sizes < pixel_size_threshold
                all_diff_vis_filter = torch.logical_and(
                    min_pixel_sizes < pixel_size_threshold,
                    gaussians.target_reso_lvl == base_reso_idx
                )
                print(f"reso {next_reso_idx} diff_vis_filter: {torch.mean(all_diff_vis_filter.float()):.3f}")

                # # show chosen points
                # prune_mask = torch.logical_not(all_diff_vis_filter)
                # gaussians.prune_points(prune_mask)
                # # visualize all chosen points
                # for cam in base_reso_cams:
                #     render_out = render(cam, gaussians, pipe, background, filter_small=filter_small,
                #             filter_large=filter_large, fade_size=fade_size)
                #     image = render_out["render"]
                #     image = torch.permute(image, (1, 2, 0))    # HWC
                #     image = image.cpu().numpy()
                #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #     cv2.imshow("image", image)
                #     key = cv2.waitKey(0)
                #     if key == ord('q'):
                #         break

                # aggregate into voxels
                gaussians.insert_large_gaussians(all_diff_vis_filter, min_pixel_sizes, next_reso_idx, scene.cameras_extent)

                # render once to update the pixel sizes
                for cam in next_reso_cams:
                    render_out = render(cam, gaussians, pipe, background, filter_small=filter_small,
                            filter_large=filter_large, fade_size=fade_size)
                    vis_filter, pixel_sizes = render_out["visibility_filter"], render_out["pixel_sizes"]
                    gaussians.update_pixel_sizes(vis_filter, pixel_sizes, next_reso_idx, iteration)
                    # for debug
                    lvl_mask = gaussians.target_reso_lvl == next_reso_idx
                    vis_lvl_mask = torch.logical_and(vis_filter, lvl_mask)
                    px_lvl_mask = torch.logical_and(pixel_sizes > 0, lvl_mask)

                # # show inserted points
                # # prune_mask = gaussians.target_reso_lvl != next_reso_idx
                # # gaussians.prune_points(prune_mask)
                # for cam in next_reso_cams:
                #     render_out = render(cam, gaussians, pipe, background, filter_small=filter_small,
                #             filter_large=filter_large, fade_size=fade_size)
                #     vis_filter, pixel_sizes = render_out["visibility_filter"], render_out["pixel_sizes"]
                #     pixel_sizes = pixel_sizes[vis_filter]
                #
                #     image = render_out["render"]
                #     image = torch.permute(image, (1, 2, 0))    # HWC
                #     image = image.cpu().numpy()
                #     # enlarge
                #     enlarge_scale = 2 ** next_reso_idx
                #     image = cv2.resize(image, (image.shape[1] * enlarge_scale, image.shape[0] * enlarge_scale), interpolation=cv2.INTER_AREA)
                #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #     cv2.imshow("image", image)
                #     cv2.setWindowTitle("image", f"reso {next_reso_idx}")
                #     key = cv2.waitKey(0)
                #     if key == ord('q'):
                #         break
                #
                # # show original reso
                # for cam in base_reso_cams:
                #     render_out = render(cam, gaussians, pipe, background, filter_small=filter_small,
                #             filter_large=filter_large, fade_size=fade_size)
                #     vis_filter, pixel_sizes = render_out["visibility_filter"], render_out["pixel_sizes"]
                #     pixel_sizes = pixel_sizes[vis_filter]
                #
                #     image = render_out["render"]
                #     image = torch.permute(image, (1, 2, 0))    # HWC
                #     image = image.cpu().numpy()
                #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #     cv2.imshow("image", image)
                #     cv2.setWindowTitle("image", f"reso {base_reso_idx}")
                #     key = cv2.waitKey(0)
                #     if key == ord('q'):
                #         break

                torch.cuda.synchronize()
                print(f"Insert large gaussians finished: {time.time() - insert_time:.3f}s")

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
                    filter_small=False, filter_large=False, fade_size=0):
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
        # output_str += f" {torch.sum(scene.gaussians.target_reso_lvl > 0)} large gs "
        for config in validation_configs:
            reso_scale = config['scale']
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                render_time = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    torch.cuda.synchronize()
                    start_time = time.time()

                    render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs,
                                            filter_small=filter_small, filter_large=filter_large, fade_size=fade_size)
                    image = torch.clamp(render_out["render"], 0.0, 1.0)

                    px = render_out["pixel_sizes"]
                    max_px = scene.gaussians.max_pixel_sizes
                    lvl = scene.gaussians.target_reso_lvl
                    valid_mask = torch.logical_and(px > 0, lvl == 4)
                    rel_px = (px / max_px)[valid_mask]
                    vis = render_out["visibility_filter"][valid_mask]
                    # print(torch.min(rel_px), torch.median(rel_px), torch.max(rel_px))

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
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                render_time /= len(config['cameras'])
                # print(f"\n[ITER {iteration}] Evaluating {config['name']} s{reso_scale:.1f}: L1 {l1_test} PSNR {psnr_test}")
                output_str += f"s{reso_scale:.1f} PSNR {psnr_test:.2f} | "
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/s{reso_scale:.1f}_loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/s{reso_scale:.1f}_loss_viewpoint - psnr", psnr_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/s{reso_scale:.1f}_loss_viewpoint - ssim", ssim_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/s{reso_scale:.1f}_loss_viewpoint - lpips", lpips_test, iteration)
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
    parser.add_argument('--grow_large', action='store_true', default=False, help='grow large gaussians')
    parser.add_argument('--insert_large', action='store_true', default=False, help='insert large gaussians')
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
             preserve_large=args.preserve_large, multi_occ=args.multi_occ, multi_dc=args.multi_dc,
             grow_large=args.grow_large, insert_large=args.insert_large)

    # All done
    print("\nTraining complete.")

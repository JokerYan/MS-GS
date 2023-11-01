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
import math

import cv2
import torch
import numpy as np
import open3d.ml.torch as ml3d

from scene.cameras import Camera
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, getWorld2View
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, reso_lvls: int = 1, multi_occ: bool = False, multi_dc: bool = False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._occ_multiplier = torch.empty(0)       # N x n_lvl x 1
        self._dc_delta = torch.empty(0)             # N x (n_lvl x 3) x 1
        self.n_lvl_occ = 4    # 2, 4, 8, 16
        self.n_lvl_dc = 4     # 2, 4, 8, 16
        self.max_radii2D = torch.empty(0)
        self.max_pixel_sizes = torch.empty(0)       # maximum pixel sizes at the highest resolution, -1 as default, N
        self.min_pixel_sizes = torch.empty(0)       # minimum pixel sizes at the highest resolution, -1 as default, N
        self.base_gaussian_mask = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)    # N x L x 1
        self.denom = torch.empty(0)                 # N x L x 1
        self.target_reso_lvl = torch.empty(0)       # N x 1, from which reso lvl the gaussian is added, default 0
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.multi_occ = multi_occ
        self.multi_dc = multi_dc
        self.reso_lvls = reso_lvls                  # L

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._occ_multiplier,
            self._dc_delta,
            self.max_radii2D,
            self.base_gaussian_mask,
            self.max_pixel_sizes,
            self.min_pixel_sizes,
            self.xyz_gradient_accum,
            self.denom,
            self.target_reso_lvl,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._occ_multiplier,
            self._dc_delta,
            self.max_radii2D,
            self.min_pixel_sizes,
            self.base_gaussian_mask,
            self.max_pixel_sizes,
            xyz_gradient_accum,
            denom,
            self.target_reso_lvl,
            opt_dict,
            self.spatial_lr_scale
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_occ_multiplier(self):
        if self.multi_occ:
            return self.opacity_activation(self._occ_multiplier)
        else:
            return self._occ_multiplier

    @property
    def get_dc_delta(self):
        return self._dc_delta

    @property
    def get_min_pixel_sizes(self):
        return self.min_pixel_sizes

    @property
    def get_max_pixel_sizes(self):
        return self.max_pixel_sizes
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_base_mask(self):
        return self.base_gaussian_mask

    @property
    def get_target_reso_lvl(self):
        return self.target_reso_lvl

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        if self.multi_occ:
            occ_multiplier = inverse_sigmoid(0.99 * torch.ones((len(fused_point_cloud), self.n_lvl_occ, 1), dtype=torch.float, device="cuda"))
        else:
            occ_multiplier = torch.ones((len(fused_point_cloud), self.n_lvl_occ, 1), dtype=torch.float, device="cuda")

        if self.multi_dc:
            dc_delta = torch.zeros((fused_point_cloud.shape[0], self.n_lvl_dc*3, 1), device="cuda")
        else:
            dc_delta = torch.zeros((fused_point_cloud.shape[0], self.n_lvl_dc*3, 1), device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._occ_multiplier = nn.Parameter(occ_multiplier.requires_grad_(True if self.multi_occ else False))
        self._dc_delta = nn.Parameter(dc_delta.requires_grad_(True if self.multi_dc else False))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_pixel_sizes = torch.ones((self.get_xyz.shape[0]), device="cuda") * -1
        self.min_pixel_sizes = torch.ones((self.get_xyz.shape[0]), device="cuda") * -1
        self.base_gaussian_mask = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool, requires_grad=False)
        self.target_reso_lvl = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.long, requires_grad=False)

    def training_setup(self, training_args):
        self.training_args = training_args
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], self.reso_lvls, 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], self.reso_lvls, 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._occ_multiplier], 'lr': training_args.opacity_lr if self.multi_occ else 0, "name": "occ_multiplier"},
            # {'params': [self._dc_delta], 'lr': training_args.feature_lr * 1e-1 if self.multi_dc else 0, "name": "dc_delta"},
            {'params': [self._occ_multiplier], 'lr': 0, "name": "occ_multiplier"},
            {'params': [self._dc_delta], 'lr': 0, "name": "dc_delta"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def mask_lvl_param_grad(self, lvl):
        # mask out the gradient of the parameters for gaussians from 'this' lvl
        mask = torch.logical_not(lvl == self.target_reso_lvl)
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                # register hook
                if len(param.shape) == 1:
                    param.register_hook(lambda grad: grad * mask)
                elif len(param.shape) == 2:
                    param.register_hook(lambda grad: grad * mask[:, None])
                elif len(param.shape) == 3:
                    param.register_hook(lambda grad: grad * mask[:, None, None])

    def start_ms_lr(self):
        # turn of base color optimization, update dc delta only
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "f_dc" \
                    or param_group["name"] == "f_rest" \
                    or param_group["name"] == "opacity" \
                    or param_group["name"] == "scaling" \
                    or param_group["name"] == "rotation":
                # param_group['lr'] = 0
                pass
            elif param_group["name"] == "occ_multiplier":
                param_group['lr'] = self.training_args.opacity_lr if self.multi_occ else 0
            elif param_group["name"] == "dc_delta":
                param_group['lr'] = self.training_args.feature_lr * 1e-1 if self.multi_dc else 0

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.n_lvl_occ):
            l.append(f'occ_multiplier_{i}')
        for i in range(self.n_lvl_dc):
            for j in range(3):
                l.append(f'dc_delta_{i}_{j}')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('base_gaussian_mask')
        l.append('max_pixel_sizes')
        l.append('min_pixel_sizes')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        occ_multiplier = self._occ_multiplier.flatten(start_dim=1).detach().cpu().numpy()
        dc_delta = self._dc_delta.flatten(start_dim=1).detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        base_gaussian_mask = self.base_gaussian_mask.detach().cpu().numpy()[:, None]

        max_pixel_sizes = self.max_pixel_sizes[:, None].detach().cpu().numpy()      # N x 1
        min_pixel_sizes = self.min_pixel_sizes[:, None].detach().cpu().numpy()      # N x 1

        dtype_full = [(attribute, 'bool') if attribute == 'base_gaussian_mask' else (attribute, 'f4')
                      for attribute in self.construct_list_of_attributes()]

        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, occ_multiplier, dc_delta,
                                     scale, rotation, base_gaussian_mask, max_pixel_sizes, min_pixel_sizes), axis=1)
        attributes = [attr for attr in attributes.transpose()]
        elements = np.core.records.fromarrays(attributes, dtype=dtype_full)
        # elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # reset only the first level
        lvl_mask = self.target_reso_lvl == 0
        opacities_new = torch.where(lvl_mask[:, None], opacities_new, self._opacity)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        # occ_multiplier_new = torch.min(self.get_occ_multiplier, torch.ones_like(self.get_occ_multiplier)*0.01)
        # optimizable_tensors = self.replace_tensor_to_optimizer(occ_multiplier_new, "occ_multiplier")
        # self._occ_multiplier = optimizable_tensors["occ_multiplier"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        occ_multiplier = np.zeros((xyz.shape[0], self.n_lvl_occ, 1))
        for i in range(self.n_lvl_occ):
            occ_multiplier[:, i, 0] = np.asarray(plydata.elements[0][f"occ_multiplier_{i}"])

        dc_delta = np.zeros((xyz.shape[0], self.n_lvl_dc*3, 1))
        for i in range(self.n_lvl_dc):
            for j in range(3):
                dc_delta[:, i*3+j, 0] = np.asarray(plydata.elements[0][f"dc_delta_{i}_{j}"])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        base_gaussian_mask = np.asarray(plydata.elements[0]["base_gaussian_mask"]).astype(np.bool)

        max_pixel_sizes = np.asarray(plydata.elements[0]["max_pixel_sizes"])
        min_pixel_sizes = np.asarray(plydata.elements[0]["min_pixel_sizes"])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities.copy(), dtype=torch.float, device="cuda").requires_grad_(True))
        self._occ_multiplier = nn.Parameter(torch.tensor(occ_multiplier.copy(), dtype=torch.float, device="cuda").requires_grad_(self.multi_occ))
        self._dc_delta = nn.Parameter(torch.tensor(dc_delta.copy(), dtype=torch.float, device="cuda").requires_grad_(self.multi_dc))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.base_gaussian_mask = torch.from_numpy(base_gaussian_mask).to(device="cuda", dtype=torch.bool)
        self.max_pixel_sizes = torch.from_numpy(max_pixel_sizes.copy()).to(device="cuda", dtype=torch.float)
        self.min_pixel_sizes = torch.from_numpy(min_pixel_sizes.copy()).to(device="cuda", dtype=torch.float)

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._occ_multiplier = optimizable_tensors["occ_multiplier"]
        self._dc_delta = optimizable_tensors["dc_delta"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]    # N x L x 1
        self.denom = self.denom[valid_points_mask]               # N x L x 1

        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_pixel_sizes = self.max_pixel_sizes[valid_points_mask]
        self.min_pixel_sizes = self.min_pixel_sizes[valid_points_mask]
        self.base_gaussian_mask = self.base_gaussian_mask[valid_points_mask]
        self.target_reso_lvl = self.target_reso_lvl[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_occ_multiplier, new_dc_delta, new_scaling, new_rotation,
                              new_target_reso_lvl, new_max_pixel_sizes, new_min_pixel_sizes, reso_lvl=0):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "occ_multiplier": new_occ_multiplier,
        "dc_delta": new_dc_delta,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._occ_multiplier = optimizable_tensors["occ_multiplier"]
        self._dc_delta = optimizable_tensors["dc_delta"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], self.reso_lvls, 1), device="cuda")    # N x L x 1
        # self.denom = torch.zeros((self.get_xyz.shape[0], self.reso_lvls, 1), device="cuda")                 # N x L x 1
        # only clear the grad at the respective reso_lvl
        self.xyz_gradient_accum[:, reso_lvl, :] = 0     # N x L x 1
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((len(new_xyz), self.reso_lvls, 1), device="cuda")], dim=0)
        self.denom[:, reso_lvl, :] = 0                  # N x L x 1
        self.denom = torch.cat([self.denom, torch.zeros((len(new_xyz), self.reso_lvls, 1), device="cuda")], dim=0)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # self.max_pixel_sizes = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # need to add zeros at the back of the current max_pixel_sizes when doing densification

        # new_max_pixel_sizes = torch.ones((len(new_xyz)), device="cuda") * -1
        self.max_pixel_sizes = torch.cat((self.max_pixel_sizes, new_max_pixel_sizes), dim=0)
        # new_min_pixel_sizes = torch.ones((len(new_xyz)), device="cuda") * -1
        self.min_pixel_sizes = torch.cat((self.min_pixel_sizes, new_min_pixel_sizes), dim=0)
        new_base_gaussian_mask = torch.zeros((len(new_xyz)), device="cuda", dtype=torch.bool)
        self.base_gaussian_mask = torch.cat((self.base_gaussian_mask, new_base_gaussian_mask), dim=0)

        self.target_reso_lvl = torch.cat((self.target_reso_lvl, new_target_reso_lvl), dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_occ_multiplier = self._occ_multiplier[selected_pts_mask].repeat(N,1,1)
        new_dc_delta = self._dc_delta[selected_pts_mask].repeat(N,1,1)
        new_target_reso_lvl = self.target_reso_lvl[selected_pts_mask].repeat(N)
        new_max_pixel_sizes = self.max_pixel_sizes[selected_pts_mask].repeat(N) / (0.8*N)
        new_min_pixel_sizes = self.min_pixel_sizes[selected_pts_mask].repeat(N) / (0.8*N)

        # # instead of pruning them, we push them to higher levels
        # self.target_reso_lvl[selected_pts_mask] = torch.clip(self.target_reso_lvl[selected_pts_mask] + 2, max=self.reso_lvls-1)
        # self.max_pixel_sizes[selected_pts_mask] = torch.ones_like(self.max_pixel_sizes[selected_pts_mask]) * -1
        # self.min_pixel_sizes[selected_pts_mask] = torch.ones_like(self.min_pixel_sizes[selected_pts_mask]) * -1

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_occ_multiplier,
                                   new_dc_delta, new_scaling, new_rotation, new_target_reso_lvl,
                                   new_max_pixel_sizes, new_min_pixel_sizes)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_occ_multiplier = self._occ_multiplier[selected_pts_mask]
        new_dc_delta = self._dc_delta[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_target_reso_lvl = self.target_reso_lvl[selected_pts_mask]
        new_max_pixel_sizes = self.max_pixel_sizes[selected_pts_mask]
        new_min_pixel_sizes = self.min_pixel_sizes[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacities, new_occ_multiplier, new_dc_delta,
                                   new_scaling, new_rotation, new_target_reso_lvl,
                                   new_max_pixel_sizes, new_min_pixel_sizes)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        reso_lvl = 0
        grads = self.xyz_gradient_accum[:, reso_lvl] / self.denom[:, reso_lvl]      # N x 1
        grads[grads.isnan()] = 0.0

        # does not densify gaussians from other lvls
        grads[self.target_reso_lvl != 0] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_size_mask = torch.logical_or(big_points_vs, big_points_ws)
            # does not prune gaussians not from lvl 0 because of screen size
            prune_size_mask = torch.logical_and(prune_size_mask, self.target_reso_lvl == 0)

            prune_mask = torch.logical_or(prune_size_mask, prune_mask)

        # debug, do not prune gaussians from other lvls at all
        prune_mask = torch.logical_and(prune_mask, self.target_reso_lvl == 0)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def grow_large_gaussians(self, grad_threshold, reso_lvl):
        grads = self.xyz_gradient_accum[:, reso_lvl] / self.denom[:, reso_lvl]      # N x 1
        grads[grads.isnan()] = 0.0

        # grad_threshold *= 3
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        grow_mult = (reso_lvl + 1) / (self.target_reso_lvl[selected_pts_mask] + 1)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self.inverse_opacity_activation(
            self.get_opacity[selected_pts_mask] / 2
            # self.get_opacity[selected_pts_mask] / grow_mult[:, None]
        )
        new_occ_multiplier = self._occ_multiplier[selected_pts_mask]
        new_dc_delta = self._dc_delta[selected_pts_mask]
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask] * 2
            # self.get_scaling[selected_pts_mask] * grow_mult[:, None]
        )
        new_rotation = self._rotation[selected_pts_mask]
        # new gaussians counts at this reso lvl
        new_target_reso_lvl = torch.ones_like(self.target_reso_lvl[selected_pts_mask]) * reso_lvl

        new_max_pixel_sizes = self.max_pixel_sizes[selected_pts_mask] * 2
        new_min_pixel_sizes = self.min_pixel_sizes[selected_pts_mask] * 2

        # print(f'grown {len(new_xyz)}/{len(self._xyz)} large gaussians at reso {reso_lvl}')
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacities, new_occ_multiplier, new_dc_delta,
                                   new_scaling, new_rotation, new_target_reso_lvl=new_target_reso_lvl,
                                   new_max_pixel_sizes=new_max_pixel_sizes, new_min_pixel_sizes=new_min_pixel_sizes,
                                   reso_lvl=reso_lvl)

    def update_pixel_sizes(self, visibility_filter, pixel_sizes, reso_lvl, iteration):
        mask = torch.logical_and(visibility_filter, self.target_reso_lvl == reso_lvl)
        if reso_lvl > 0 and torch.any(self.target_reso_lvl == reso_lvl):
            assert True

        # prevent the min/max pixel sizes is outdated
        self.min_pixel_sizes[mask] = torch.clip(self.min_pixel_sizes[mask] * 1.05, -1)
        self.max_pixel_sizes[mask] = self.max_pixel_sizes[mask] * 0.95

        self.max_pixel_sizes[mask] = torch.max(self.max_pixel_sizes[mask], pixel_sizes[mask])

        self.min_pixel_sizes[mask] = torch.where(
            self.min_pixel_sizes[mask] < 0,  # if not initialized
            torch.where(
                pixel_sizes[mask] > 0,  # pixel size is valid
                pixel_sizes[mask],
                self.min_pixel_sizes[mask]
            ),
            torch.where(  # if initialized
                pixel_sizes[mask] > 0,  # pixel size is valid
                torch.min(self.min_pixel_sizes[mask], pixel_sizes[mask]),
                self.min_pixel_sizes[mask]
            ))

    def prune_small_points(self):
        raise NotImplementedError("prune_small_points needs some adjustment after the large gaussian growth")
        if torch.max(self.max_pixel_sizes) > 0:
            prune_mask = self.max_pixel_sizes < 1.0
            self.prune_points(prune_mask)
            torch.cuda.empty_cache()
        else:
            print("max pixel sizes too small for purning")
        self.max_pixel_sizes = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def add_densification_stats(self, viewspace_point_tensor, update_filter, reso_lvl=0):
        self.xyz_gradient_accum[:, reso_lvl][update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)  # N x 1
        self.denom[:, reso_lvl][update_filter] += 1

    def update_base_gaussian_mask(self, visibility_filter):
        # only call this when training at the lowest resolution
        self.base_gaussian_mask = torch.logical_or(self.base_gaussian_mask, visibility_filter)

    def add_large_gaussian(self, camera: Camera, render_results):
        rgb = render_results["render"]
        acc_pixel_size = render_results["acc_pixel_size"]
        depth = render_results["depth"]

        # small pixel size mask
        small_pixel_size_mask = acc_pixel_size < 3.0        # threshold slightly smaller than target size
        # cv2.imshow("small_pixel_size_mask", small_pixel_size_mask.cpu().numpy().astype(np.float32))
        # cv2.waitKey(0)

        # avg pool to k x k
        k = 4
        small_pixel_size_mask = small_pixel_size_mask.float()
        small_pixel_size_mask = small_pixel_size_mask[None, None, :, :]             # (1, 1, H, W)
        small_pixel_size_mask = F.avg_pool2d(small_pixel_size_mask, kernel_size=k, stride=k, padding=0, ceil_mode=True)
        small_pixel_size_mask = small_pixel_size_mask.squeeze(0).squeeze(0)         # (H, W)
        small_pixel_size_mask = small_pixel_size_mask >= 0.75

        small_depth = depth[None, None, :, :]             # (1, 1, H, W)
        small_depth = F.avg_pool2d(small_depth, kernel_size=k, stride=k, padding=0, ceil_mode=True)
        small_depth = small_depth.squeeze(0).squeeze(0)         # (H, W)

        small_rgb = rgb[None, :, :, :]             # (1, 3, H, W)
        small_rgb = F.avg_pool2d(small_rgb, kernel_size=k, stride=k, padding=0, ceil_mode=True)
        small_rgb = small_rgb.squeeze(0)           # (3, H, W)
        small_rgb = small_rgb.permute(1, 2, 0)      # (H, W, 3)

        # convert mask to coordinates
        ys, xs = torch.where(small_pixel_size_mask)
        zs = small_depth[ys, xs]
        rgb_pt = small_rgb[ys, xs, :]           # (N, 3)
        if len(ys) == 0:
            return False
        ys = ys * k + k / 2 - 0.5       # convert back to orignal size coordinates
        xs = xs * k + k / 2 - 0.5

        # back project to 3d
        fovx = camera.FoVx
        fovy = camera.FoVy
        H, W = camera.image_height, camera.image_width
        cx = W / 2 - 0.5
        cy = H / 2 - 0.5
        half_x = math.tan(fovx / 2)
        half_y = math.tan(fovy / 2)
        xs = (xs - cx) / cx * half_x * zs
        ys = (ys - cy) / cy * half_y * zs
        coord = torch.stack([xs, ys, zs], dim=-1)
        coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)     # homogeneous coordinates, N x 4

        # calculate scale of each new gaussian
        scale_x = half_x / (W / 2) * (k / 2)
        scale_y = half_y / (H / 2) * (k / 2)
        scale_x = scale_x * zs      # N
        scale_y = scale_y * zs      # N
        scale_z = torch.mean(torch.stack([scale_x, scale_y], dim=-1), dim=-1)     # N
        new_scaling = torch.stack([scale_x, scale_y, scale_z], dim=-1)            # N x 3
        new_scaling = self.scaling_inverse_activation(new_scaling)

        # print("scale", torch.max(new_scaling), torch.min(new_scaling), torch.mean(self._scaling), torch.max(self._scaling), torch.min(self._scaling))
        # print("depth", torch.min(zs), torch.max(zs))

        # transform to world space
        w2c = getWorld2View(camera.R, camera.T)
        w2c = torch.from_numpy(w2c).to(device=coord.device)
        c2w = torch.linalg.inv(w2c)                 # 4 x 4
        coord = c2w @ coord.transpose(0, 1)         # 4 x N
        coord = coord.transpose(0, 1)[:, :3]        # N x 3

        # calculate attributes of gaussians
        N = len(coord)
        new_feature_dc = RGB2SH(rgb_pt[:, None, :])     # (N, 1, 3)
        new_feature_rest = torch.zeros([N, (self.max_sh_degree + 1) ** 2 - 1, 3], device=coord.device)     # (N, 8, 3)
        new_opacity = torch.ones([N, 1], device=coord.device) * 0.5       # (N, 1)
        new_rotation = torch.zeros([N, 4], device=coord.device)       # (N, 4)

        # max_idx = torch.argmax(torch.max(new_scaling, dim=1).values, dim=0)
        # print("scale", new_scaling[max_idx], "color", new_feature_dc[max_idx])

        # add new gaussian
        self.densification_postfix(coord, new_feature_dc, new_feature_rest, new_opacity, new_scaling, new_rotation)
        return True

    def insert_large_gaussians(self, mask, cur_min_pixel_sizes, reso_lvl, scene_extent):
        # scale positions from (-inf, inf) to (-2, 2)
        # everything within the scene extent are scaled linearly to (-1, 1)
        # everything beyond that are scaled to (-2, 2) inversely, by (2 - 1/x)
        rel_pos = self._xyz[mask] / scene_extent
        rel_pos = torch.where(rel_pos > 1, 2 - 1/rel_pos, rel_pos)
        rel_pos = rel_pos.cpu()

        # aggregate into voxels
        N = len(self._xyz)      # total_number of points
        # voxel_reso = 0.02 * (1.5 ** reso_lvl)
        voxel_reso = 0.02 * (reso_lvl / 4)
        # voxel_reso = 0.02
        voxel_pooling = ml3d.layers.VoxelPooling(position_fn='center', feature_fn='average')
        # voxel_pooling = ml3d.layers.VoxelPooling(position_fn='center', feature_fn='nearest_neighbor')
        # voxel_pooling_max = ml3d.layers.VoxelPooling(position_fn='center', feature_fn='max')

        voxel_xyz = voxel_pooling(rel_pos, self._xyz.reshape([N, -1])[mask].cpu(), voxel_reso).pooled_features.cuda()
        voxel_feature_dc = voxel_pooling(rel_pos, self._features_dc.reshape([N, -1])[mask].cpu(), voxel_reso).pooled_features.cuda()
        voxel_feature_rest = voxel_pooling(rel_pos, self._features_rest.reshape([N, -1])[mask].cpu(), voxel_reso).pooled_features.cuda()
        voxel_opacity = voxel_pooling(rel_pos, self._opacity.reshape([N, -1])[mask].cpu(), voxel_reso).pooled_features.cuda()
        voxel_occ_multiplier = voxel_pooling(rel_pos, self._occ_multiplier.reshape([N, -1])[mask].cpu(), voxel_reso).pooled_features.cuda()
        voxel_dc_delta = voxel_pooling(rel_pos, self._dc_delta.reshape([N, -1])[mask].cpu(), voxel_reso).pooled_features.cuda()
        voxel_rotation = voxel_pooling(rel_pos, self._rotation.reshape([N, -1])[mask].cpu(), voxel_reso).pooled_features.cuda()
        voxel_max_pixel_sizes = voxel_pooling(rel_pos, self.max_pixel_sizes.reshape([N, -1])[mask].cpu(), voxel_reso).pooled_features.cuda()
        voxel_min_pixel_sizes = voxel_pooling(rel_pos, self.min_pixel_sizes.reshape([N, -1])[mask].cpu(), voxel_reso).pooled_features.cuda()
        voxel_scaling = voxel_pooling(rel_pos, self._scaling.reshape([N, -1])[mask].cpu(), voxel_reso).pooled_features.cuda()
        voxel_cur_min_pixel_sizes = voxel_pooling(rel_pos, cur_min_pixel_sizes.reshape([N, -1])[mask].cpu(), voxel_reso).pooled_features.cuda()

        # reshape back
        M = len(voxel_xyz)    # number of voxels
        voxel_xyz = voxel_xyz.reshape([M, *self._xyz.shape[1:]])
        voxel_feature_dc = voxel_feature_dc.reshape([M, *self._features_dc.shape[1:]])
        voxel_feature_rest = voxel_feature_rest.reshape([M, *self._features_rest.shape[1:]])
        voxel_opacity = voxel_opacity.reshape([M, *self._opacity.shape[1:]])
        voxel_occ_multiplier = voxel_occ_multiplier.reshape([M, *self._occ_multiplier.shape[1:]])
        voxel_dc_delta = voxel_dc_delta.reshape([M, *self._dc_delta.shape[1:]])
        voxel_rotation = voxel_rotation.reshape([M, *self._rotation.shape[1:]])
        voxel_max_pixel_sizes = voxel_max_pixel_sizes.reshape([M, *self.max_pixel_sizes.shape[1:]])
        voxel_min_pixel_sizes = voxel_min_pixel_sizes.reshape([M, *self.min_pixel_sizes.shape[1:]])
        voxel_scaling = voxel_scaling.reshape([M, *self._scaling.shape[1:]])    # (M, 3)
        voxel_cur_min_pixel_sizes = voxel_cur_min_pixel_sizes.reshape([M, 1])   # (M, 1)

        # increase size and reduce opacity
        voxel_cur_min_pixel_sizes = torch.clip(voxel_cur_min_pixel_sizes, min=0.1, max=2.0)  # clip to 0.1~2.0
        voxel_scaling_factor = 2.0 / voxel_cur_min_pixel_sizes
        voxel_scaling = self.scaling_inverse_activation(self.scaling_activation(voxel_scaling) * voxel_scaling_factor)
        voxel_opacity = self.inverse_opacity_activation(self.opacity_activation(voxel_opacity) / 1)
        # voxel_max_pixel_sizes = voxel_max_pixel_sizes * 4
        # voxel_min_pixel_sizes = voxel_min_pixel_sizes * 4
        voxel_max_pixel_sizes = torch.ones((M), device="cuda") * -1
        voxel_min_pixel_sizes = torch.ones((M), device="cuda") * -1

        self.densification_postfix(voxel_xyz, voxel_feature_dc, voxel_feature_rest,
                                   voxel_opacity, voxel_occ_multiplier, voxel_dc_delta,
                                   voxel_scaling, voxel_rotation, reso_lvl=reso_lvl,
                                   new_max_pixel_sizes=voxel_max_pixel_sizes, new_min_pixel_sizes=voxel_min_pixel_sizes,
                                   new_target_reso_lvl=torch.ones_like(voxel_max_pixel_sizes) * reso_lvl,
                                   )
        print(f"Inserted {len(voxel_xyz)} large gaussians at reso {2**reso_lvl}")

    def filter_center(self, max_dist, train=False):
        # filter the gassians to leave only the gaussians that are close to the center
        # this is only for the purpose of visualization
        dist = torch.norm(self.get_xyz, dim=-1)
        mask = dist < max_dist

        self._xyz = self._xyz[mask]
        self._features_dc = self._features_dc[mask]
        self._features_rest = self._features_rest[mask]
        self._opacity = self._opacity[mask]
        self._occ_multiplier = self._occ_multiplier[mask]
        self._dc_delta = self._dc_delta[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]

        self.max_pixel_sizes = self.max_pixel_sizes[mask]
        self.min_pixel_sizes = self.min_pixel_sizes[mask]
        self.base_gaussian_mask = self.base_gaussian_mask[mask]

        if train:
            self.max_radii2D = self.max_radii2D[mask]
            self.xyz_gradient_accum = self.xyz_gradient_accum[mask]    # N x L x 1
            self.denom = self.denom[mask]               # N x L x 1
            self.target_reso_lvl = self.target_reso_lvl[mask]
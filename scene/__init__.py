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
import pickle
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")) \
                or os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # cache cameras
        # check if the resolutions are all 2^n
        is_normal_scales = True
        for scale in resolution_scales:
            if scale not in [2**i for i in range(len(resolution_scales))]:
                is_normal_scales = False
                break
        if is_normal_scales:
            camera_cache_name = f"cameras_{len(resolution_scales)}.pkl"
        else:
            camera_cache_name = f"cameras_({'-'.join([str(reso) for reso in resolution_scales])}).pkl"
        camera_cache_path = os.path.join(args.source_path, camera_cache_name)
        if os.path.exists(camera_cache_path):
            print("Loading cameras from cache")
            camera_data = pickle.load(open(camera_cache_path, "rb"))
            # scene_info = camera_data["scene_info"]
            self.train_cameras = camera_data["train_cameras"]
            self.test_cameras = camera_data["test_cameras"]
            print("Camera cache loaded")
        else:
            if shuffle:
                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            camera_data = {
                # "scene_info": scene_info,
                "train_cameras": self.train_cameras,
                "test_cameras": self.test_cameras
            }
            pickle.dump(camera_data, open(camera_cache_path, "wb"))
            print(f"Saved cameras to cache {camera_cache_path}")

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
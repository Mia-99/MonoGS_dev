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
import sys
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
import uuid
from tqdm import tqdm
import wandb
from random import randint
import numpy as np
import copy

import torch
import torch.multiprocessing as mp


from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import Scene
from gaussian_splatting.scene.gaussian_model_GS import GaussianModel
from gaussian_splatting.utils.general_utils import safe_state
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_splatting.utils.graphics_utils import BasicPointCloud



from utils.pose_utils import update_pose


from gui import gui_utils, sfm_gui
from utils.multiprocessing_utils import FakeQueue, clone_obj


from submodules.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
import cv2
import matplotlib
import PIL

import pycolmap
import pathlib



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False





class DepthAnything:

    def __init__(self, encoder = 'vitb') -> None:
        # encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }

        self.model = DepthAnythingV2(**self.model_configs[encoder])
        self.model.load_state_dict(torch.load(f'submodules/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = self.model.to('cuda').eval()

    def eval(self, raw_img):
        depth = self.model.infer_image(raw_img) # HxW raw depth map in numpy
        # print(f"depth = {depth}")
        return depth
    
    @staticmethod
    # candidate colormaps: 'nipy_spectral', 'Spectral_r', 'turbo', 'plasma'
    def depth2image (depth, colormap=None):
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        if colormap is not None:
            cmap = matplotlib.colormaps.get_cmap(colormap)
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        return depth






class SFM(mp.Process):

    def __init__(self, pipe = None, q_main2vis = None, q_vis2main = None, use_gui = True, viewpoint_stack = None, gaussians = None, opt = None, cameras_extent = None) -> None:
        self.pipe = pipe
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.use_gui = use_gui

        self.viewpoint_stack = viewpoint_stack
        self.gaussians = gaussians
        self.opt = opt

        self.background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        self.rgb_boundary_threshold = 0.01

        self.pause = False


        self.require_calibration = True
        self.allow_lens_distortion = True        
        self.add_calib_noise_iter = -1
        self.start_calib_iter = 200

        self.cameras_extent = cameras_extent

        self.depth_anything = DepthAnything()
        self.depth_scale = 10



    def updateCalibration(self, flag=False):
        focal = torch.zeros(1, device=self.viewpoint_stack[0].device)
        kappa = torch.zeros(1, device=self.viewpoint_stack[0].device)
        for viewpoint_cam in self.viewpoint_stack:
            focal += viewpoint_cam.cam_focal_delta
            kappa += viewpoint_cam.cam_kappa_delta
            # Don't forget to zero this vector every iteration. Otherwise the gradient will accumulate over iterations
            viewpoint_cam.cam_focal_delta.data.fill_(0)
            viewpoint_cam.cam_kappa_delta.data.fill_(0)

        if flag == True:
            focal = focal.cpu().numpy()[0]
            kappa = kappa.cpu().numpy()[0]
            print(f"calib update: focal {focal}, kappa {kappa}")
            for viewpoint_cam in self.viewpoint_stack:
                viewpoint_cam.fx += focal
                viewpoint_cam.fy += viewpoint_cam.aspect_ratio * focal
                viewpoint_cam.kappa += kappa




    def optimize (self):

        cam_cnt = 0
        if self.use_gui:
            self.q_main2vis.put(
                gui_utils.GaussianPacket(
                    gaussians=self.gaussians,
                    current_frame=clone_obj(self.viewpoint_stack[cam_cnt]),
                    keyframes=copy.deepcopy(self.viewpoint_stack),
                )
            )
            time.sleep(5)


        sfm_gui.Log("start SfM optimization")

        first_iter = 0

        self.gaussians.training_setup(self.opt)

   
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)


        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
        first_iter += 1

        # optimize pose
        opt_params = []

        for viewpoint_cam in self.viewpoint_stack:
            opt_params.append(
                {
                    "params": [viewpoint_cam.cam_rot_delta],
                    "lr": 0.003,
                    "name": "rot_{}".format(viewpoint_cam.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint_cam.cam_trans_delta],
                    "lr": 0.001,
                    "name": "trans_{}".format(viewpoint_cam.uid),
                }
            )
            if self.require_calibration:
                opt_params.append(
                    {
                        "params": [viewpoint_cam.cam_focal_delta],
                        "lr": 1.0,
                        "name": "calibration_f_{}".format(viewpoint_cam.uid),
                    }
                )
                if self.allow_lens_distortion:
                    opt_params.append(
                        {
                            "params": [viewpoint_cam.cam_kappa_delta],
                            "lr": 0.0001,
                            "name": "calibration_k_{}".format(viewpoint_cam.uid),
                        }
                    )
        pose_optimizer = torch.optim.Adam(opt_params)
        pose_optimizer.zero_grad()



        for iteration in range(first_iter, self.opt.iterations+1):
            

            # interaction with gui interface Pause/Resume
            if not self.q_vis2main.empty():
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause            
                while self.pause:
                    if self.q_vis2main.empty():
                            continue
                    else:
                        data_vis2main = self.q_vis2main.get()
                        self.pause = data_vis2main.flag_pause



            iter_start.record()


            # add noise to calibration to test the robustness
            if iteration == self.add_calib_noise_iter:
                for viewpoint_cam in self.viewpoint_stack:
                    focal = 650
                    viewpoint_cam.fx = focal
                    viewpoint_cam.fy = viewpoint_cam.aspect_ratio * focal
                    viewpoint_cam.kappa = 0.0
                for param_group in pose_optimizer.param_groups:
                    if "calibration_f_" in param_group["name"]:
                        param_group["lr"] = 2.0

                if self.use_gui:
                    cam_cnt = (cam_cnt+1) % len(self.viewpoint_stack)
                    depth = np.zeros((self.viewpoint_stack[0].image_height, self.viewpoint_stack[0].image_width))
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=self.gaussians,
                            keyframes=copy.deepcopy(self.viewpoint_stack),
                            current_frame=clone_obj(self.viewpoint_stack[cam_cnt]),
                            gtcolor=self.viewpoint_stack[cam_cnt].original_image,
                            gtdepth=depth,
                        )
                    )
                    time.sleep(0.001)
                time.sleep(3)


 
            # upapte learning rate for calibration parameters
            if iteration == self.add_calib_noise_iter + 100:
                for param_group in pose_optimizer.param_groups:
                    if "calibration_f_" in param_group["name"]:
                        param_group["lr"] *= 0.1
            if iteration == self.add_calib_noise_iter + 200:
                for param_group in pose_optimizer.param_groups:
                    if "calibration_f_" in param_group["name"]:
                        param_group["lr"] *= 0.1
            if iteration == self.add_calib_noise_iter + 300:
                for param_group in pose_optimizer.param_groups:
                    if "calibration_f_" in param_group["name"]:
                        param_group["lr"] *= 0.1
                    if "calibration_k_" in param_group["name"]:
                        param_group["lr"] *= 0.1



            self.gaussians.update_learning_rate(iteration)


            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()



            loss = 0.0
            for k in range(len(self.viewpoint_stack)):

                viewpoint_cam = self.viewpoint_stack[k]

                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background,
                                    scaling_modifier=1.0,
                                    override_color=None,
                                    mask=None,)

                image, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["opacity"], render_pkg["n_touched"]                   

                if self.use_gui and iteration % 10 == 0 and k == (cam_cnt+1) % len(self.viewpoint_stack):
                    depth = opacity.squeeze().cpu().detach().numpy()

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                # define mask using background color threshold
                # print(f"gt_image.shape = {gt_image.shape}")
                # print(f"gt_image.sum(dim=0).shape = {gt_image.sum(dim=0).shape}")
                # print(f"gt_image.sum(dim=0) = {gt_image.sum(dim=0)}")   
                # print(f"mask = { (gt_image.sum(dim=0) > self.rgb_boundary_threshold) }")                
                mask = (gt_image.sum(dim=0) > self.rgb_boundary_threshold)
                # mask = opacity
                Ll1 = l1_loss(image, gt_image)
                # Ll1 = l1_loss(image*mask, gt_image*mask)
                loss = loss + (1.0 - self.opt.lambda_dssim) * Ll1   #+ self.opt.lambda_dssim * (1.0 - ssim(image*mask, gt_image*mask))
        
            loss.backward()



            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == self.opt.iterations:
                    progress_bar.close()

                # Densification
                if True and iteration < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        sfm_gui.Log("Densify and Prune Gaussians", tag="SFM")
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.cameras_extent, size_threshold)
                    
                    if iteration % self.opt.opacity_reset_interval == 0:
                        sfm_gui.Log("Reset opacity of all Gaussians", tag="SFM")
                        self.gaussians.reset_opacity()


                # Optimizer step
                if iteration > 0 and iteration < self.opt.iterations:
                    pose_optimizer.step()
                    self.updateCalibration(flag = ( self.require_calibration and iteration >= self.start_calib_iter ))
                    for viewpoint_cam in self.viewpoint_stack:
                        if viewpoint_cam.uid != 0:
                            update_pose(viewpoint_cam)
                    pose_optimizer.zero_grad()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)

                calib_profiling = self.require_calibration and iteration >= self.start_calib_iter and iteration < self.start_calib_iter + 30
                if self.use_gui and (iteration % 10 == 0 or calib_profiling):
                    # depth = np.zeros((self.viewpoint_stack[0].image_height, self.viewpoint_stack[0].image_width))
                    cam_cnt = (cam_cnt+1) % len(self.viewpoint_stack)
                    img = self.viewpoint_stack[cam_cnt].original_image.cpu().squeeze(0).numpy()
                    # convert (3, H, W)  --- > (H, W, 3)
                    # coorect color order BGR2RGB
                    cv_img = img.transpose(1, 2, 0)
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    # use depth prediction from a Neural network                    
                    depth = self.depth_anything.eval(cv_img) / self.depth_scale
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=self.gaussians,
                            keyframes=copy.deepcopy(self.viewpoint_stack),
                            current_frame=clone_obj(self.viewpoint_stack[cam_cnt]),
                            gtcolor=self.viewpoint_stack[cam_cnt].original_image,
                            gtdepth=depth,
                        )
                    )
                    time.sleep(0.001)
                    if calib_profiling:
                        time.sleep(1.0)

                    
        sfm_gui.Log(f"SfM optimization complete with {iteration} iterations.")

        if self.use_gui:
            self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))  
            time.sleep(3.0)







if __name__ == "__main__":

    mp.set_start_method('spawn')


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
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)


    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)


    print(opt.__dict__)

    print(pipe.__dict__)



    opt.iterations = 500
    opt.densification_interval = 50
    opt.opacity_reset_interval = 350
    opt.densify_from_iter = 49
    opt.densify_until_iter = 200
    opt.densify_grad_threshold = 0.0002



    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    cameras_extent = scene.cameras_extent


    N = 3

    viewpoint_stack = scene.getTrainCameras()
    while len(viewpoint_stack) > N:
        viewpoint_stack.pop(-1)
    sfm_gui.Log(f"cameras used: {len(scene.getTrainCameras())}")

    viewpoint_stack = scene.getTrainCameras().copy()

    # in original 3DGS, R is transposed in colmap reader and later inverted in getWorld2View2
    # in this code, getWorld2View2 don't transpose R
    for cam in viewpoint_stack:
        Rt = torch.transpose(cam.R, 0, 1)
        cam.R = Rt



    torch.autograd.set_detect_anomaly(args.detect_anomaly)



    ## visualization
    use_gui = True
    q_main2vis = mp.Queue() if use_gui else FakeQueue()
    q_vis2main = mp.Queue() if use_gui else FakeQueue()


    if use_gui:
        bg_color = [0, 0, 0]
        params_gui = gui_utils.ParamsGUI(
            pipe=pipe,
            background=torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
            gaussians=GaussianModel(dataset.sh_degree),
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )
        gui_process = mp.Process(target=sfm_gui.run, args=(params_gui,))
        gui_process.start()
        time.sleep(1)


    print(f"Run with image W: { viewpoint_stack[0].image_width },  H: { viewpoint_stack[0].image_height }")

    sfm = SFM(pipe, q_main2vis, q_vis2main, use_gui, viewpoint_stack, gaussians, opt, cameras_extent)
    sfm.add_calib_noise_iter = 250
    sfm.start_calib_iter = 250
    sfm_process = mp.Process(target=sfm.optimize)
    sfm_process.start()

  
    torch.cuda.synchronize()


    if use_gui:
        gui_process.join()
        sfm_gui.Log("GUI Stopped and joined the main thread", tag="GUI")



    print(q_main2vis.empty())
    print(q_vis2main.empty())

    sfm_process.join()
    sfm_gui.Log("Finished", tag="SfM")

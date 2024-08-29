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
import torch.optim.lr_scheduler as lr_scheduler


from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import Scene
from gaussian_splatting.scene.gaussian_model_GS import GaussianModel
from gaussian_splatting.utils.general_utils import safe_state
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_splatting.utils.graphics_utils import BasicPointCloud

from gaussian_splatting.utils.general_utils import helper as lr_helper


from utils.pose_utils import update_pose


from gui import gui_utils, sfm_gui
from utils.multiprocessing_utils import FakeQueue, clone_obj


from depth_anything import DepthAnything




try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False





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
        focal_delta = torch.zeros(1, device=self.viewpoint_stack[0].device)
        kappa_delta = torch.zeros(1, device=self.viewpoint_stack[0].device)
        for viewpoint_cam in self.viewpoint_stack:
            focal_delta += viewpoint_cam.cam_focal_delta
            kappa_delta += viewpoint_cam.cam_kappa_delta
            # Don't forget to zero this vector every iteration. Otherwise the gradient will accumulate over iterations
            viewpoint_cam.cam_focal_delta.data.fill_(0)
            viewpoint_cam.cam_kappa_delta.data.fill_(0)

        if flag == True:
            focal_delta = focal_delta.cpu().numpy()[0]
            kappa_delta = kappa_delta.cpu().numpy()[0]
            for viewpoint_cam in self.viewpoint_stack:
                viewpoint_cam.fx += focal_delta
                viewpoint_cam.fy += viewpoint_cam.aspect_ratio * focal_delta
                viewpoint_cam.kappa += kappa_delta
            print(f"update calibration: focal_delta = {focal_delta},  kappa_delta = {kappa_delta}")



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
            time.sleep(3)


        sfm_gui.Log("start SfM optimization")

        first_iter = 0

        self.gaussians.training_setup(self.opt)

   
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)


        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
        first_iter += 1



        # optimize pose
        pose_opt_params = []
        for viewpoint_cam in self.viewpoint_stack:
            pose_opt_params.append(
                {
                    "params": [viewpoint_cam.cam_rot_delta],
                    "lr": 0.003,
                    "name": "rot_{}".format(viewpoint_cam.uid),
                }
            )
            pose_opt_params.append(
                {
                    "params": [viewpoint_cam.cam_trans_delta],
                    "lr": 0.001,
                    "name": "trans_{}".format(viewpoint_cam.uid),
                }
            )
        pose_optimizer = torch.optim.Adam(pose_opt_params)
        pose_optimizer.zero_grad()



        # optimize calibration
        calib_opt_params = []
        for viewpoint_cam in self.viewpoint_stack:
            calib_opt_params.append(
                {
                    "params": [viewpoint_cam.cam_focal_delta],
                    "lr": 0.1,
                    "name": "calibration_f_{}".format(viewpoint_cam.uid),
                }
            )
            if self.allow_lens_distortion:
                calib_opt_params.append(
                    {
                        "params": [viewpoint_cam.cam_kappa_delta],
                        "lr": 0.001,
                        "name": "calibration_k_{}".format(viewpoint_cam.uid),
                    }
                )
        calib_optimizer = torch.optim.NAdam(calib_opt_params)
        calib_optimizer.zero_grad()



        for iteration in range(first_iter, self.opt.iterations+1):
            

            # interaction with gui interface Pause/Resume
            if not self.q_vis2main.empty():
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause            
                while self.pause:
                    if self.q_vis2main.empty():
                            time.sleep(0.01)
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
                for param_group in calib_optimizer.param_groups:
                    if "calibration_f_" in param_group["name"]:
                        param_group["lr"] = 2.0  # start from a large leraning rate as we modify focal abruptly.

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



            if iteration > self.start_calib_iter and (iteration - self.start_calib_iter) % 100 == 0:
                for param_group in calib_optimizer.param_groups:
                    if "calibration_f_" in param_group["name"]:
                        lr = param_group["lr"]
                        param_group["lr"] = 0.1 * lr if lr > 0.01 else lr


            self.gaussians.update_learning_rate(iteration)


            forzen_states = ( iteration > self.start_calib_iter and iteration < self.start_calib_iter + 50 )

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

                # Loss
                gt_image = viewpoint_cam.original_image.cuda() 
                mask = (gt_image.sum(dim=0) > self.rgb_boundary_threshold)
                # mask = opacity
                # Ll1 = l1_loss(image, gt_image)
                Ll1 = l1_loss(image*mask, gt_image*mask)
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
                if self.require_calibration and iteration > 0 and iteration < self.opt.iterations and iteration >= self.start_calib_iter:
                    calib_optimizer.step()
                    self.updateCalibration(flag = True)
                    calib_optimizer.zero_grad()

                # Optimizer step
                if iteration > 0 and iteration < self.opt.iterations and not forzen_states:
                    pose_optimizer.step()
                    for viewpoint_cam in self.viewpoint_stack:
                        if viewpoint_cam.uid != 0:
                            update_pose(viewpoint_cam)
                    pose_optimizer.zero_grad()

                # Optimizer step
                if iteration > 0 and iteration < self.opt.iterations and not forzen_states:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)
                

                if self.use_gui and (iteration % 10 == 0 or forzen_states):
                    # depth = np.zeros((self.viewpoint_stack[0].image_height, self.viewpoint_stack[0].image_width))
                    cam_cnt = (cam_cnt+1) % len(self.viewpoint_stack)
                    # img = (self.viewpoint_stack[cam_cnt].original_image*255).cpu().squeeze(0).numpy()
                    # cv_img = img.transpose(1, 2, 0)
                    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    cv_img = (self.viewpoint_stack[cam_cnt].original_image*255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    # use depth prediction from a Neural network                    
                    depth = self.depth_anything.eval(cv_img)
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
                    if forzen_states:
                        time.sleep(0.1)

                    
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


    # print(opt.__dict__)
    # print(pipe.__dict__)



    opt.iterations = 1000
    opt.densification_interval = 50
    opt.opacity_reset_interval = 350
    opt.densify_from_iter = 49
    opt.densify_until_iter = 750
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
        bg_color = [0.0, 0.0, 0.0]
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
    sfm.add_calib_noise_iter = 50
    sfm.start_calib_iter = 50
    sfm_process = mp.Process(target=sfm.optimize)
    sfm_process.start()

  
    torch.cuda.synchronize()


    if use_gui:
        gui_process.join()
        sfm_gui.Log("GUI Stopped and joined the main thread", tag="GUI")



    sfm_process.join()
    sfm_gui.Log("Finished", tag="SfM")

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

import torch
import torch.multiprocessing as mp


from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import Scene
from gaussian_splatting.scene.gaussian_model_GS import GaussianModel
from gaussian_splatting.utils.general_utils import safe_state
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams


from utils.pose_utils import update_pose


from gui import gui_utils, sfm_gui
from utils.multiprocessing_utils import FakeQueue, clone_obj



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False





class SFM:

    def __init__(self, pipe = None, q_main2vis = None, q_vis2main = None, use_gui = True) -> None:
        self.pipe = pipe
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.use_gui = use_gui
        self.require_calibration = True



    def updateCalibration(self):

        focal = torch.zeros(1, device=self.viewpoint_stack[0].device)
        kappa = torch.zeros(1, device=self.viewpoint_stack[0].device)

        for viewpoint_cam in self.viewpoint_stack:

            focal = focal + viewpoint_cam.cam_focal_delta
            kappa = kappa + viewpoint_cam.cam_kappa_delta

            viewpoint_cam.cam_focal_delta.data.fill_(0)
            viewpoint_cam.cam_kappa_delta.data.fill_(0)

        focal = focal.cpu().numpy()[0]
        kappa = kappa.cpu().numpy()[0]

        print(f"update focal = {focal}, update kappa = {kappa}")

        # print(f"calib update: focal {focal}, kappa {kappa}")

        for viewpoint_cam in self.viewpoint_stack:
            viewpoint_cam.fx = viewpoint_cam.fx + focal
            viewpoint_cam.fy = viewpoint_cam.fy + viewpoint_cam.aspect_ratio * focal
            viewpoint_cam.kappa = viewpoint_cam.kappa + kappa

        pass


    def run (self, viewpoint_stack, gaussians, opt, bg_color = [1, 1, 1]):


        self.viewpoint_stack = viewpoint_stack
        self.gaussians = gaussians
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        print(f"Run with image W: { viewpoint_stack[0].image_width },  H: { viewpoint_stack[0].image_height }")
        
        

        cam_cnt = 0
        if self.use_gui:
            self.q_main2vis.put(
                gui_utils.GaussianPacket(
                    gaussians=self.gaussians,
                    current_frame=self.viewpoint_stack[cam_cnt],
                    keyframes=self.viewpoint_stack,
                )
            )
            time.sleep(1.5)


        print("start optimization")

        first_iter = 0

        self.gaussians.training_setup(opt)

   
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)


        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
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
            opt_params.append(
                {
                    "params": [viewpoint_cam.exposure_a],
                    "lr": 0.01,
                    "name": "exposure_a_{}".format(viewpoint_cam.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint_cam.exposure_b],
                    "lr": 0.01,
                    "name": "exposure_b_{}".format(viewpoint_cam.uid),
                }
            )
            if self.require_calibration:
                opt_params.append(
                    {
                        "params": [viewpoint_cam.cam_focal_delta],
                        "lr": 0.100,
                        "name": "calibration_f_{}".format(viewpoint_cam.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint_cam.cam_kappa_delta],
                        "lr": 0.01,
                        "name": "calibration_k_{}".format(viewpoint_cam.uid),
                    }
                )            
        pose_optimizer = torch.optim.Adam(opt_params)
        pose_optimizer.zero_grad()



        for iteration in range(first_iter, opt.iterations+1):


            iter_start.record()

            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 50 == 0:
                self.gaussians.oneupSHdegree()


            bg = torch.rand((3), device="cuda") if opt.random_background else self.background

            loss = 0.0

            for k in range(len(self.viewpoint_stack)):

                viewpoint_cam = self.viewpoint_stack[k]

                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, bg)

                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)

                loss = loss + (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))    
        
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

                # Densification
                if True and iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        sfm_gui.Log("Densify and Prune Gaussians", tag="SFM")
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        sfm_gui.Log("Reset opacity of all Gaussians", tag="SFM")
                        self.gaussians.reset_opacity()


                # Optimizer step
                if iteration > 0 and iteration < opt.iterations:
                    pose_optimizer.step()
                    if self.require_calibration:
                        self.updateCalibration()
                    for viewpoint_cam in self.viewpoint_stack:
                        if viewpoint_cam.uid != 0:
                            update_pose(viewpoint_cam)
                    pose_optimizer.zero_grad()

                # Optimizer step
                if iteration < opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)


                if self.use_gui and iteration % 10 == 0:
                    cam_cnt += 1
                    cam_cnt = cam_cnt % len(self.viewpoint_stack)
                    depth = np.zeros((viewpoint_stack[0].image_height, viewpoint_stack[0].image_width))
                    q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=self.gaussians,  #clone_obj(gaussians)
                            keyframes=self.viewpoint_stack,
                            current_frame=self.viewpoint_stack[cam_cnt],
                            gtcolor=self.viewpoint_stack[cam_cnt].original_image,
                            gtdepth=depth,
                        )
                    )
                    # time.sleep(0.001)
                    # print("")

        if self.use_gui:
            self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))  
            time.sleep(1.0)

        print("\nTraining complete.")






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



    opt.iterations = 1000
    opt.densification_interval = 50
    opt.opacity_reset_interval = 200
    opt.densify_from_iter = 49
    opt.densify_until_iter = 3000
    opt.densify_grad_threshold = 0.0002



    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)


    N = 3

    viewpoint_stack = scene.getTrainCameras()
    while len(viewpoint_stack) > N:
        viewpoint_stack.pop()
    sfm_gui.Log(f"cameras used: {len(scene.getTrainCameras())}")

    viewpoint_stack = scene.getTrainCameras().copy()

    # for viewpoint_cam in viewpoint_stack:
    #     focal = 800
    #     viewpoint_cam.fx = focal
    #     viewpoint_cam.fy = viewpoint_cam.aspect_ratio * focal
    #     viewpoint_cam.kappa = 0.1



    torch.autograd.set_detect_anomaly(args.detect_anomaly)



    ## visualization
    use_gui = True
    q_main2vis = mp.Queue() if use_gui else FakeQueue()
    q_vis2main = mp.Queue() if use_gui else FakeQueue()


    if use_gui:
        bg_color = [1, 1, 1]
        params_gui = gui_utils.ParamsGUI(
            pipe=pipe,
            background=torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
            gaussians=GaussianModel(dataset.sh_degree),
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )
        gui_process = mp.Process(target=sfm_gui.run, args=(params_gui,))
        gui_process.start()


    torch.cuda.synchronize()


    SFM(pipe, q_main2vis, q_vis2main, use_gui).run(viewpoint_stack, gaussians, opt)
  

    if use_gui:
        gui_process.join()
        sfm_gui.Log("GUI Stopped and joined the main thread", tag="GUI")


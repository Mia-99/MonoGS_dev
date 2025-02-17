
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


from optimizers import CalibrationOptimizer, PoseOptimizer, LineDetection, lr_exp_decay_helper

from gaussian_scale_space import image_conv_gaussian_separable

from matplotlib import pyplot as plt

import pathlib
import rich

from gaussian_viewer import Viewer, create_gaussians_gl


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False




    
def print_viewpoint_stack(viewpoint_stack, prefix="Camera"):
    for viewpoint_cam in viewpoint_stack:
        uid = viewpoint_cam.uid
        calib_id = viewpoint_cam.calib_id
        fx = viewpoint_cam.fx
        fy = viewpoint_cam.fy
        kappa = viewpoint_cam.kappa     
        CC = viewpoint_cam.camera_center.cpu().numpy()
        exposure_a = viewpoint_cam.exposure_a.data.item()
        exposure_b = viewpoint_cam.exposure_b.data.item()
        rich.print(f"[bold blue]{prefix}[/bold blue] uid: [{uid}]: calib_id: {calib_id}. fx: {fx:.3f}, fy: {fy:.3f}, kappa: {kappa:.6f}. cam_center: ({CC[0]:.3f}, {CC[1]:.3f}, {CC[2]:.3f}), exposure: (a: {exposure_a:.5f}, b: {exposure_b:.5f})")



class CameraResectioning(mp.Process):


    def __init__(self, pipe = None, use_gui = True, viewpoint_stack = None, gaussians = None, opt = None, cameras_extent = None) -> None:
        self.pipe = pipe
        self.use_gui = use_gui

        self.viewpoint_stack = viewpoint_stack   # list of cameras
        self.gaussians = gaussians   # fixed in camera resectioning
        self.opt = opt

        self.background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        self.rgb_boundary_threshold = 0.01

        self.pause = False
        

        self.require_calibration = True
        self.allow_lens_distortion = True

        self.focal_reference = None

        self.cameras_extent = cameras_extent

        self.calibration_optimizer = None
        self.pose_optimizer = None
        self.calib_safe_guard = False

        self.gaussian_scale_t = 10
        self.image_margin_mask = None


        self.MODULE_TEST_CALIBRATION = False
        self.add_calib_noise_iter = -1


        self.q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        self.q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        if self.use_gui:
            bg_color = [0.0, 0.0, 0.0]
            params_gui = gui_utils.ParamsGUI(
                pipe=pipe,
                background=torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
                gaussians=self.gaussians if self.gaussians is not None else GaussianModel(0),
                q_main2vis=self.q_main2vis,
                q_vis2main=self.q_vis2main,
            )
            self.gui_process = mp.Process(target=sfm_gui.run, args=(params_gui,))
            self.gui_process.start()
            time.sleep(3)



    def push_to_gui (self, cam_cnt):
        depth = np.zeros((self.viewpoint_stack[0].image_height, self.viewpoint_stack[0].image_width))
        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                gaussians=clone_obj(self.gaussians),
                keyframes=self.viewpoint_stack,
                current_frame=self.viewpoint_stack[cam_cnt],
                gtcolor=self.viewpoint_stack[cam_cnt].original_image,
                gtdepth=depth,
            )
        )
        time.sleep(0.001)
    

    def read_gui_ctrl (self):
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


    def show_rendered_images (self):
        for viewpoint in self.viewpoint_stack:
            render_pkg = render(viewpoint, self.gaussians, self.pipe, self.background,
                                scaling_modifier=1.0,
                                override_color=None,
                                mask=None,)
            image, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["opacity"], render_pkg["n_touched"]
            # convert torch tensor to opencv image
            rgb = torch.clamp(image, min=0, max=1.0) * 255
            rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            plt.imshow(rgb)
            plt.title(f"view uid: {viewpoint.uid}", fontweight ="bold") 
            plt.show()


    """

    Optimization subroutines

    
    """
    def compute_loss_one_view (self, viewpoint, use_scale_space = False, use_SSIM = False):
        # Loss function
        loss = 0.0

        render_pkg = render(viewpoint, self.gaussians, self.pipe, self.background,
                            scaling_modifier=1.0,
                            override_color=None,
                            mask=None,)

        image, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["opacity"], render_pkg["n_touched"]                   

        # Loss
        gt_image = viewpoint.original_image.cuda() 
        mask = (gt_image.sum(dim=0) > self.rgb_boundary_threshold)
        # mask = opacity
        # Ll1 = l1_loss(image, gt_image)

        # Gaussian scale space for focal length calibration
        if use_scale_space and self.gaussian_scale_t > 0.5:
            image_scale_t = image_conv_gaussian_separable(image, sigma=self.gaussian_scale_t, epsilon=0.01) * mask
            gt_image_scale_t = image_conv_gaussian_separable(gt_image, sigma=self.gaussian_scale_t, epsilon=0.01) * mask
        else:
            image_scale_t = image #* mask
            gt_image_scale_t = gt_image #* mask

        # huber_loss_function = torch.nn.HuberLoss(reduction = 'mean', delta = 1.0)
        huber_loss_function = torch.nn.SmoothL1Loss(reduction = 'mean', beta = 0.0)
        loss += (1.0 - self.opt.lambda_dssim) * huber_loss_function(image_scale_t, gt_image_scale_t)

        # Ll1 = l1_loss(image*mask, gt_image*mask)  
        # loss += (1.0 - self.opt.lambda_dssim) * Ll1

        # enable SSIM loss when a good intialial reconstruction is attained
        if use_SSIM:
            loss += self.opt.lambda_dssim * (1.0 - ssim(image*mask, gt_image*mask))

        return loss, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched
    
    

    def optimize_one_step (self, iteration, update_Gaussian = False, update_pose = False, update_calibration = False,  use_scale_space = False, use_ssim_loss = False, densify_prune = False, reset_opacity = False):

        self.gaussian_iter += 1
        self.gaussians.update_learning_rate(self.gaussian_iter)

        for viewpoint in self.viewpoint_stack:

            # FORWARD
            loss, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched = self.compute_loss_one_view ( viewpoint, use_scale_space = use_scale_space,  use_SSIM = use_ssim_loss )            

            # BACKWARD
            loss.backward()

            with torch.no_grad():

                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if densify_prune:
                    sfm_gui.Log("Densify and Prune Gaussians", tag="SFM")
                    size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.cameras_extent, size_threshold)
                
                if reset_opacity:
                    sfm_gui.Log("Reset opacity of all Gaussians", tag="SFM")
                    self.gaussians.reset_opacity()

            # calibration step            
            if update_calibration:

                for viewpoint in self.viewpoint_stack:
                    rich.print(f"[bold yellow]After loss.backward: [/bold yellow]{viewpoint.cam_focal_delta.grad=}")
                self.calibration_optimizer.focal_step()

                if self.allow_lens_distortion:
                    for viewpoint in self.viewpoint_stack:
                        rich.print(f"[bold red]After loss.backward: [/bold red]{viewpoint.cam_kappa_delta.grad=}")
                    self.calibration_optimizer.kappa_step()

            # pose step
            if update_pose:
                self.pose_optimizer.step()
            
            # Gaussian step
            update_Gaussian = False
            if update_Gaussian:
                self.gaussians.optimizer.step()

            self.calibration_optimizer.zero_grad() # clear gradient every iteration
            self.pose_optimizer.zero_grad() # clear gradient every iteration
            self.gaussians.optimizer.zero_grad(set_to_none = True) # clear gradient every iteration




    def optimize (self, max_iters = 1000, set_focal_error=None):

        _, h, w = self.viewpoint_stack[0].original_image.shape
        self.image_margin_mask = torch.zeros(h, w).cuda()
        band_with = int(1.0 * self.gaussian_scale_t)
        self.image_margin_mask[band_with:-band_with,  band_with:-band_with] = 1.0
        if self.focal_reference is None:
            self.focal_reference = np.sqrt(h*h + w*w)/2

        if self.calibration_optimizer is None:            
            self.calibration_optimizer = CalibrationOptimizer(self.viewpoint_stack, focal_reference = self.focal_reference, focal_optimizer_type = "Adam")
            self.calibration_optimizer.update_focal_learning_rate (lr = 0.03) # 0.1 also works
            self.calib_safe_guard = False

        if self.pose_optimizer is None:
            self.pose_optimizer = PoseOptimizer(self.viewpoint_stack)

        cam_cnt = 0
        if self.use_gui:
            self.push_to_gui(cam_cnt)
            time.sleep(1.5)

        self.gaussians.training_setup(self.opt)

        sfm_gui.Log("start SfM optimization")


        '''
        Optimization
        '''
        progress_bar = tqdm(range(1, max_iters+1), desc="Phase1: Training progress")
        cam_cnt = 0
        for iteration in range(0, max_iters):
            self.read_gui_ctrl()
            densify_prune = iteration and (iteration % 20 ==0)
            reset_opacity = iteration and (iteration % 300 ==0)
            self.optimize_one_step (iteration,
                                    update_Gaussian = False,
                                    update_pose = True,
                                    update_calibration = True,
                                    use_scale_space = True,
                                    densify_prune = densify_prune,
                                    reset_opacity = reset_opacity
                                    )
            print_viewpoint_stack(self.viewpoint_stack, prefix="Camera")
            if self.use_gui and (iteration % 5 == 0):
                self.push_to_gui(cam_cnt)
                cam_cnt = (cam_cnt+1) % len(self.viewpoint_stack)
            if iteration % 10 == 0:
                with torch.no_grad():
                    loss_log = self.compute_loss (use_scale_space = False,  use_SSIM = False )
                progress_bar.set_postfix({"Loss": f"{loss_log:.{7}f}"})
                progress_bar.update(10)
        progress_bar.close()


        if set_focal_error is not None:
            ''' For debug and test
            '''
            noise_fx = set_focal_error
            for viewpoint in self.viewpoint_stack:
                focal = viewpoint.fx + noise_fx
                viewpoint.fx = focal
                viewpoint.fy = viewpoint.aspect_ratio * focal
            rich.print(f"[bold red][Notice]: old fx {focal - noise_fx} ====> new fx {focal}.  Noise added {noise_fx}  [/bold red]")

        sfm_gui.Log(f"SfM optimization complete.")
        torch.cuda.synchronize()

        self.close()


    def close(self):
        torch.cuda.synchronize()
        if self.use_gui:
            self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            self.gui_process.join()
            sfm_gui.Log("GUI Stopped and joined the main thread", tag="GUI")
        time.sleep(0.01)
    

    def eval_data(self):
        W2C_arr, fx_arr, fy_arr, kappa_arr, rendered_images, captured_images, error_images = [], [], [], [], [], [], []
        for viewpoint in self.viewpoint_stack:
            R = viewpoint.R.detach().cpu().numpy()
            T = viewpoint.T.detach().cpu().numpy()
            fx = viewpoint.fx
            fy = viewpoint.fy
            kappa = viewpoint.kappa
            original_image = viewpoint.original_image.detach().cpu()
            rendering = render(viewpoint, self.gaussians, self.pipe, self.background,
                                scaling_modifier=1.0,
                                override_color=None,
                                mask=None,)["render"].detach().cpu()
            render_image = torch.clamp(rendering, 0.0, 1.0)
            W2C = np.eye(4)
            W2C[0:3, 0:3] = R
            W2C[0:3, 3] = T
            W2C_arr.append (W2C)
            fx_arr.append (fx)
            fy_arr.append (fy)
            kappa_arr.append (kappa)
            rendered_images.append(render_image)
            captured_images.append(original_image)
            error_images.append(torch.abs(rendering - original_image))
            # print(error_images[-1])
        return (W2C_arr, fx_arr, fy_arr, kappa_arr, rendered_images, captured_images, error_images) 
    

    @staticmethod
    def tensor2rgb (image):
        '''
            OpenCV stores images in BGR order instead of RGB
            plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        '''
        rgb = torch.clamp(image, min=0, max=1.0) * 255
        rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
        return rgb
    




def main():

    camera_file_path = "/hdd/3DGS/bicycle/cameras.json"
    point_cloud_file_path = "/hdd/3DGS/bicycle/point_cloud/iteration_7000/point_cloud.ply"


    gaussians = GaussianModel(sh_degree=0)

    cam_infos = read_camera_json (camera_file_path)
    gaussians.load_ply(point_cloud_file_path)



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



    opt.iterations = 200
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
    use_gui = False
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

    sfm = CameraResectioning(pipe, q_main2vis, q_vis2main, use_gui, viewpoint_stack, gaussians, opt, cameras_extent)

    sfm.MODULE_TEST_CALIBRATION = True
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

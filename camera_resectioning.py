
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


from utils.camera_utils import Camera
from utils.pose_utils import update_pose


from gui import gui_utils, sfm_gui
from utils.multiprocessing_utils import FakeQueue, clone_obj


from optimizers import CalibrationOptimizer, PoseOptimizer, LineDetection, lr_exp_decay_helper

from gaussian_scale_space import image_conv_gaussian_separable

from matplotlib import pyplot as plt

import pathlib
import rich
import json

from gaussian_viewer import Viewer, create_gaussians_gl


from colmap_utils.gaussian_splatting_utils import assemble_3DGS_cameras_from_3DGS_JSON_file

from matplot_utils import annotate_image

import cv2


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


    def __init__(self, pipe = None, use_gui = True, viewpoint_stack = None, gaussians = None, opt = None) -> None:
        self.pipe = pipe
        self.use_gui = use_gui

        self.viewpoint_stack = viewpoint_stack   # list of cameras
        self.gaussians = gaussians   # fixed in camera resectioning
        self.opt = opt

        self.gaussians.optimizer = None # Do NOT optimize Gaussian

        self.background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        self.rgb_boundary_threshold = 0.01

        self.pause = False
        

        self.require_calibration = True
        self.allow_lens_distortion = True

        self.focal_reference = None

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



    def set_viewpoint_calibration (self, view_id=0, delta_focal=0.0, delta_kappa=0.0):
        assert ( view_id >= 0 and view_id < len(self.viewpoint_stack) ), f"view_id={view_id} out of range!"
        viewpoint = self.viewpoint_stack[view_id]

        focal = viewpoint.fx + delta_focal
        kappa = viewpoint.kappa + delta_kappa
        
        viewpoint.fx = focal
        viewpoint.fy = focal * viewpoint.aspect_ratio
        viewpoint.kappa = kappa

        # save ground-truth values
        viewpoint.fx_init = viewpoint.fx
        viewpoint.fy_init = viewpoint.fy
        viewpoint.kappa_init = viewpoint.kappa
        viewpoint.R_gt = viewpoint.R
        viewpoint.T_gt = viewpoint.T

        # render a distorted image
        render_pkg = render(viewpoint, self.gaussians, self.pipe, self.background,
                            scaling_modifier=1.0,
                            override_color=None,
                            mask=None,)
        image = render_pkg["render"]
        viewpoint.original_image = image.data.clone()
        return viewpoint



    def optimize (self, view_id = 0, max_iters = 1000, set_focal_error=None, set_kappa_error=None, update_pose=False, update_calibration = True):
        assert ( view_id >= 0 and view_id < len(self.viewpoint_stack) ), f"view_id={view_id} out of range!"
        viewpoint = self.viewpoint_stack[view_id]
        if viewpoint.original_image is None:            
            return
        
        _, h, w = viewpoint.original_image.shape
        self.image_margin_mask = torch.zeros(h, w).cuda()
        band_with = int(1.0 * self.gaussian_scale_t)
        self.image_margin_mask[band_with:-band_with,  band_with:-band_with] = 1.0
        if self.focal_reference is None:
            self.focal_reference = np.sqrt(h*h + w*w)/2

        if self.calibration_optimizer is None:            
            self.calibration_optimizer = CalibrationOptimizer([ viewpoint ], focal_reference = self.focal_reference, focal_optimizer_type = "Adam")
            self.calibration_optimizer.update_focal_learning_rate (lr = 0.02) # 0.1 also works
            self.calibration_optimizer.update_kappa_learning_rate (lr = 0.001)
            self.calib_safe_guard = False

        if self.pose_optimizer is None:
            self.pose_optimizer = PoseOptimizer([ viewpoint ])

        """
        use noisy initial value
        """
        if set_focal_error is not None:
            noise_fx = set_focal_error
            focal = viewpoint.fx + noise_fx
            viewpoint.fx = focal
            viewpoint.fy = focal * viewpoint.aspect_ratio
            rich.print(f"[bold red][Notice]: old fx {focal - noise_fx} ====> new fx {focal}.  Noise added {noise_fx}  [/bold red]")

        if set_kappa_error is not None:
            noise_kappa = set_kappa_error
            kappa = viewpoint.kappa + noise_kappa
            viewpoint.kappa = kappa
            rich.print(f"[bold red][Notice]: old kappa {kappa - noise_kappa} ====> new kappa {kappa}.  Noise added {noise_kappa}  [/bold red]")


        if self.use_gui:
            self.push_to_gui(view_id)
            time.sleep(1.5)

        sfm_gui.Log("start Camera Resectioning Optimization")


        '''
        Optimization
        '''
        for iteration in range(0, max_iters):
            self.read_gui_ctrl()

            use_scale_space = False,
            use_ssim_loss = False

            # FORWARD
            loss = self.compute_loss_one_view ( viewpoint, use_scale_space = use_scale_space,  use_SSIM = use_ssim_loss )            
            # BACKWARD
            loss.backward()
            with torch.no_grad():
                # calibration step            
                if update_calibration:
                    # rich.print(f"[bold yellow]After loss.backward: [/bold yellow]{viewpoint.cam_focal_delta.grad=}")
                    self.calibration_optimizer.focal_step()
                    if self.allow_lens_distortion:
                        rich.print(f"[bold red]After loss.backward: [/bold red]{viewpoint.cam_kappa_delta.grad=}")
                        self.calibration_optimizer.kappa_step()
                # pose step
                if update_pose:
                    self.pose_optimizer.step()
                self.calibration_optimizer.zero_grad() # clear gradient every iteration
                self.pose_optimizer.zero_grad() # clear gradient every iteration

            print_viewpoint_stack([ viewpoint ], prefix=f"Camera {view_id}")

            if self.use_gui and (iteration % 5 == 0):
                self.push_to_gui(view_id)
                time.sleep(0.5)

        sfm_gui.Log(f"optimization complete.")
        torch.cuda.synchronize()

        self.close()



    def show_rendered_images (self, view_id = None, save_to_dir=None, annotate=True):
        # plt.rcParams["font.family"] = "Arial"
        # plt.rcParams["font.family"] = "Times New Roman"
        csfont = {'fontname':'Times New Roman'}
        for id, viewpoint in enumerate(self.viewpoint_stack):
            if (view_id is not None) and id != view_id:
                continue
            render_pkg = render(viewpoint, self.gaussians, self.pipe, self.background,
                                scaling_modifier=1.0,
                                override_color=None,
                                mask=None,)
            image, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["opacity"], render_pkg["n_touched"]
            # convert torch tensor to opencv image
            rgb = self.tensor2rgb(image)
            mytext = f"view uid: {viewpoint.uid}, fx: {viewpoint.fx: .2f}, fy: {viewpoint.fy: .2f}, k: {viewpoint.kappa: .6f}" if annotate else None
            fig, ax, _ = annotate_image(rgb, cmap=None, mytext = mytext)
            if save_to_dir is not None:
                post_str = f"_k{viewpoint.kappa: .6f}"
                plt.savefig(os.path.join(save_to_dir, "view"+str(view_id)+post_str+'.png'), bbox_inches='tight', pad_inches=0)
                plt.close()
                time.sleep(0.01)

        plt.show(block=False)





    """

    subroutines    

    """

    def push_to_gui (self, cam_cnt):
        depth = np.zeros((self.viewpoint_stack[cam_cnt].image_height, self.viewpoint_stack[cam_cnt].image_width))
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
        # mask = mask * opacity

        # Gaussian scale space for focal length calibration
        if use_scale_space and self.gaussian_scale_t > 0.5:
            image_scale_t = image_conv_gaussian_separable(image, sigma=self.gaussian_scale_t, epsilon=0.01)
            gt_image_scale_t = image_conv_gaussian_separable(gt_image, sigma=self.gaussian_scale_t, epsilon=0.01)
        else:
            image_scale_t = image
            gt_image_scale_t = gt_image

        Ll1 = l1_loss(image_scale_t*mask, gt_image_scale_t*mask)
        loss += (1.0 - self.opt.lambda_dssim) * Ll1

        # enable SSIM loss when a good intialial reconstruction is attained
        if use_SSIM:
            loss += self.opt.lambda_dssim * (1.0 - ssim(image*mask, gt_image*mask))

        return loss
    
 


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
    



def distort_by_opencv (image_file, kappa, fx, fy):
    # https://stackoverflow.com/a/68706787/4926757
    def invert_map(F):
        sh = (F.shape[0], F.shape[1])
        I = np.zeros_like(F)
        I[:,:,1], I[:,:,0] = np.indices(sh)
        P = np.copy(I)
        for i in range(10):
            P += I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        return P

    k_1 = kappa

    img = cv2.imread(image_file, cv2.IMREAD_COLOR)

    h, w, c = img.shape[0], img.shape[1], img.shape[2]


    dist_coeffs = np.array([k_1, 0, 0, 0, 0])  # (k1, k2, p1, p2, k3)
    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = fx
    camera_matrix[1, 1] = fy
    camera_matrix[0, 2] = (w-1)/2
    camera_matrix[1, 2] = (h-1)/2
    new_camera_matrix = camera_matrix.copy()

    # Compute "Undistort" maps:  
    mapxy, _ = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), m1type=cv2.CV_32FC2)

    # Invert the maps    
    inv_mapxy = invert_map(mapxy)
    inv_mapx = inv_mapxy[:, :, 0]
    inv_mapy = inv_mapxy[:, :, 1]

    # Use the inverted maps
    out_img = cv2.remap(img, inv_mapx, inv_mapy, cv2.INTER_LINEAR)

    cv2.imwrite('view_cv_out_img_' + f"k{kappa:.6f}" + '.png', out_img)


if __name__ == "__main__":

    mp.set_start_method('spawn')

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--data_dir", default="/hdd/3DGS/playroom")
    parser.add_argument("--data_iter_num", default=7000) # 30000

    args = parser.parse_args(sys.argv[1:])

    base_dir = args.data_dir
    iter_num = args.data_iter_num

    print(args.data_dir)
    print(args.data_iter_num)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)



    """
    read 3DGS rendering output
    """

    camera_file_path = os.path.join(base_dir, "cameras.json")
    point_cloud_file_path = os.path.join(base_dir, "point_cloud/iteration_"+str(iter_num)+"/point_cloud.ply")

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(point_cloud_file_path)

    viewpoint_stack = assemble_3DGS_cameras_from_3DGS_JSON_file (camera_file_path)


    PnP = CameraResectioning(pipe = pipe, use_gui = False, viewpoint_stack = viewpoint_stack, gaussians = gaussians, opt = opt)


    view_id = 0

    delta_focal = 0.0
    fx = PnP.viewpoint_stack[view_id].fx
    fy = PnP.viewpoint_stack[view_id].fy
    

    delta_kappa = 0.0
    PnP.set_viewpoint_calibration(view_id, delta_focal=delta_focal, delta_kappa=delta_kappa)
    PnP.show_rendered_images(view_id, save_to_dir=".", annotate=False)


    delta_kappa = -0.000045
    PnP.set_viewpoint_calibration(view_id, delta_focal=delta_focal, delta_kappa=delta_kappa)
    PnP.show_rendered_images(view_id, save_to_dir=".", annotate=True)


    delta_kappa = 0.000045 + 0.55
    PnP.set_viewpoint_calibration(view_id, delta_focal=delta_focal, delta_kappa=delta_kappa)
    PnP.show_rendered_images(view_id, save_to_dir=".", annotate=True)


    delta_kappa = 0.000045 - 0.55 - 0.25
    PnP.set_viewpoint_calibration(view_id, delta_focal=delta_focal, delta_kappa=delta_kappa)
    PnP.show_rendered_images(view_id, save_to_dir=".", annotate=True)


    distort_by_opencv (image_file="view0_k 0.000000.png", kappa=0.55, fx=fx, fy=fy)
    distort_by_opencv (image_file="view0_k 0.000000.png", kappa=-0.25, fx=fx, fy=fy)


    # PnP.optimize (view_id = 0, max_iters = 1000,
    #               set_focal_error=-delta_focal, set_kappa_error=-delta_kappa,
    #               update_pose=False, update_calibration = True)





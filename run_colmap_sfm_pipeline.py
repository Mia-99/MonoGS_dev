
import time
from datetime import datetime
import sys, os

from gaussian_splatting.gaussian_renderer import render
import torch
import torch.multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.scene.gaussian_model_GS import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gui import gui_utils, sfm_gui, slam_gui

import time
from argparse import ArgumentParser, Namespace
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams

from gaussian_splatting.utils.general_utils import safe_state
from utils.multiprocessing_utils import FakeQueue, clone_obj

from PIL import Image
from gaussian_splatting.utils.general_utils import PILtoTorch

import open3d as o3d



from sfm import SFM
from colmap import ColMap
from colmap import assemble_3DGS_cameras

from gaussian_viewer import Viewer, create_gaussians_gl
from utils_cali.eval_cali_utils import eval_ate, save_cali, save_gaussians_class, eval_rendering
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
import cv2
import json
# from depth_anything import DepthAnything
# def init_dense_pcd_from_network (viewpoint_stack, reconstruction: ColMap, num_points = 20000):

#     pcd_downsample_factor = viewpoint_stack[0].image_height * viewpoint_stack[0].image_width * len(viewpoint_stack) / num_points

#     DA = DepthAnything()

#     positions = None
#     colors = None

#     for cam in viewpoint_stack:

#         sparse_depth_stack = reconstruction.getSparseDepthFromImage(image_id = cam.uid, downsample_scale = downsample_scale )
#         rgb_raw = (cam.original_image *255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

#         # use depth prediction from a Neural network
#         disp_raw = DA.eval(rgb_raw)
#         depth_raw = 10.0 / disp_raw  # depth = (focal * baseline) / disparity

#         # depth_rect = DA.correct_depth_from_sparse_points (depth=depth_raw, uv_depth_stack=sparse_depth_stack)

#         scale = DA.estimateScaleFactor(depth=depth_raw, uv_depth_stack=sparse_depth_stack)
#         depth_rect = depth_raw * scale
#         print(f"depth scale correction = {scale}, rgb_raw.shape = {rgb_raw.shape} depth_raw.shape = {depth_raw.shape}, depth_rect.shape = {depth_rect.shape}")


#         if False:
#             plt.rcParams["figure.figsize"] = (15, 6)
#             fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
#             ax1.imshow(rgb_raw)
#             ax2.imshow(depth_raw)
#             ax3.imshow(depth_rect)
#             plt.show()


#         # RGB-D image to pcd in world frame
#         rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
#         depth = o3d.geometry.Image(depth_rect.astype(np.float32))
#         new_xyz, new_rgb = GaussianModel.create_pcd_from_image_and_depth(cam, rgb, depth, downsample_factor = pcd_downsample_factor)
        
#         positions = np.concatenate((positions, new_xyz), axis=0) if positions is not None else new_xyz
#         colors = np.concatenate((colors, new_rgb), axis=0) if colors is not None else new_rgb

#     return positions, colors




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


    opt.iterations = 100
    opt.densification_interval = 30
    opt.opacity_reset_interval = 200
    opt.densify_from_iter = 49
    opt.densify_until_iter = 2000
    opt.densify_grad_threshold = 0.0002


    """ DATASET URL
    
    https://colmap.github.io/datasets.html#datasets
    
    https://cvg-data.inf.ethz.ch/

    """

    data_url = "https://cvg-data.inf.ethz.ch/local-feature-evaluation-schoenberger2017/South-Building.zip"
    '''
    ground_truth (not provided)
    128 images of the “South” building at UNC Chapel Hill. The images are taken with the same camera, kindly provided by Christopher Zach.
    '''

    data_url = "https://cvg-data.inf.ethz.ch/local-feature-evaluation-schoenberger2017/Strecha-Herzjesu.zip"
    image_dir = "/hdd/sfm/Strecha-Herzjesu/Herzjesu/images"
    '''
    ground_truth calibration:
        2759.48 0 1520.69
        0 2764.16 1006.81
    '''
    
    data_url = "https://cvg-data.inf.ethz.ch/local-feature-evaluation-schoenberger2017/Strecha-Fountain.zip"
    image_dir = "/datasets/Strecha-Herzjesu/Herzjesu/images"
    '''
    ground_truth calibration:
        2759.48 0 1520.69
        0 2764.16 1006.81
    '''


    use_pcd_from_depth_prediction = False


    # perform colmap reconstruction
    reconstruction = ColMap(image_dir)

    # extract reconstruction information: 1. posedCameras, 2. 3Dpointcloud
    downsample_scale = 2**2
    viewpoint_stack, scale_info = assemble_3DGS_cameras(reconstruction,  downsample_scale = downsample_scale,  use_same_calib = True)
        
    print(f"scale_info = {scale_info}")
    cameras_extent = scale_info["radius"]

    # initialize 3D Gaussians from sparse Colmap output
    gaussians = GaussianModel(sh_degree=0)
    gaussians.spatial_lr_scale = cameras_extent
    
    positions, colors = reconstruction.getPointCloud()
    pcd = BasicPointCloud(points=positions, colors=colors, normals=None)
    gaussians.create_from_pcd(pcd, cameras_extent)
    gaussians.training_setup(opt)



    print(f"Run with image W: { viewpoint_stack[0].image_width },  H: { viewpoint_stack[0].image_height }")

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    ## visualization
    use_gui = True
    sfm = SFM(pipe, use_gui, viewpoint_stack, gaussians, opt, cameras_extent)
    sfm.optimize()
    # sfm.close()
    
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = "./results/sfm/" + image_dir.split("/")[-2] + "/" + current_datetime
    # read R_gt, t_gt
    # image_dir
    # R_gt, t_gt = read_gt(image_dir)

    def load_gt(image_directory):
        print(f"Loading ground truth camera parameters from {image_directory}")
        # remove the '/images' in the directory
        parent_directory = os.path.dirname(image_directory)
        groundtruth_directory = os.path.join(parent_directory, 'groundtruth')
        camera_files = [f for f in os.listdir(groundtruth_directory) if f.endswith('.camera')]
        print(f"camera_files = {camera_files}")
        camera_files.sort()  # Ensure numerical order

        all_camera_params = []  # List to store all camera parameters
        fxs = []
        fys = []
        R_gts = []
        T_gts = []

        for filename in camera_files:
            filepath = os.path.join(groundtruth_directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()

                # Parsing intrinsic matrix
                intrinsic = np.array([list(map(float, lines[i].strip().split())) for i in range(3)])
                fx = intrinsic[0, 0]
                fxs.append(fx)
                fy = intrinsic[1, 1]
                fys.append(fy)

                # Parsing extrinsic parameters (rotation matrix and translation vector)
                rotation = np.array([list(map(float, lines[i].strip().split())) for i in range(4, 7)])
                R_gts.append(torch.tensor(rotation, dtype=torch.float32, device=torch.device('cuda')))
                translation = np.array(list(map(float, lines[7].strip().split())))
                T_gts.append(torch.tensor(translation, dtype=torch.float32, device=torch.device('cuda')))
                
                
                # Image dimensions
                dimensions = list(map(int, lines[8].strip().split()))

                # Store in a dictionary
                camera_params = {
                    'intrinsic': intrinsic,
                    'rotation': rotation,
                    'translation': translation,
                    'dimensions': dimensions
                }
                all_camera_params.append(camera_params)
        
        return R_gts, T_gts, fxs, fys
    # load_gt(image_dir)
    R_gts, T_gts, fxs, fys = load_gt(image_dir)
    # print( sfm.viewpoint_stack)
    # print(len(R_gts))
    for i in range(len(sfm.viewpoint_stack)):
        sfm.viewpoint_stack[i].R_gt = R_gts[i]
        sfm.viewpoint_stack[i].T_gt = T_gts[i]
        sfm.viewpoint_stack[i].fx_init = fxs[i]/downsample_scale
        sfm.viewpoint_stack[i].fy_init = fys[i]/downsample_scale
        
    eval_ate(sfm.viewpoint_stack, [i for i in range(len(sfm.viewpoint_stack))], save_dir=path, iterations=0, final=True, monocular=True)
    save_cali(save_dir=path, frames=sfm.viewpoint_stack, kf_indices=[i for i in range(len(sfm.viewpoint_stack))], N_frames=None)
    save_gaussians_class(save_dir=path, gaussians=sfm.gaussians)
    
    def save_rendering(viewpoints, gaussians, kf_indices, pipeline_params, save_path):
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        dir = os.path.join(save_path, 'rendering')
        os.makedirs(os.path.join(dir, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(dir, 'gt'), exist_ok=True)
        img_pred, img_gt, saved_frame_idx = [], [], []
        psnr_array, ssim_array, lpips_array = [], [], []
        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to("cuda")
        for i in range(len(kf_indices)):
            idx = kf_indices[i]
            # gt_image format:
            # torch.Size([3, 600, 800])
            # torch.float32
            # cuda:0
            # let viewpoints[idx].original_image in cuda:0 device
            gt_image = viewpoints[idx].original_image.to("cuda:0")
            viewpoint = viewpoints[idx]
            # viewpoint.compute_grad_mask(self.config)


            # TODO: add pipeline_params and background

            rendering = render(viewpoint, gaussians, pipeline_params, background)["render"]
            image = torch.clamp(rendering, 0.0, 1.0)
            # save image
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

            # print(gt.shape)  # Should show a tuple of (height, width, channels) for color or (height, width) for grayscale
            # print(gt.dtype)  # Should typically be 'uint8' for image data
            # print(os.path.join(dir, f'gt/image_gt_{idx}.png'))
            cv2.imwrite(os.path.join(dir, f'pred/image_pred_{idx}.png'), pred)
            state = cv2.imwrite(os.path.join(dir, f'gt/image_gt_{idx}.png'), gt)


            mask = gt_image > 0
            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
            lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

            psnr_array.append(psnr_score.item())
            ssim_array.append(ssim_score.item())
            lpips_array.append(lpips_score.item())
            # path/pred/image_pred_{i}.png

        print('mean psnr:', np.mean(psnr_array))
        print('mean ssim:', np.mean(ssim_array))
        print('mean lpips:', np.mean(lpips_array))
        # write mean psnr, ssim, lpips to a file
        with open(os.path.join(dir, 'psnr_final_result.json'), 'w') as file:
            json.dump({
                'mean_psnr': np.mean(psnr_array),
                'mean_ssim': np.mean(ssim_array),
                'mean_lpips': np.mean(lpips_array),
                'psnr_array': psnr_array,
                'ssim_array': ssim_array,
                'lpips_array': lpips_array
            }, file)
        return rendering
    
    save_rendering(sfm.viewpoint_stack, sfm.gaussians, [i for i in range(len(sfm.viewpoint_stack))], pipe, path)
    # eval_rendering, save_gaussians_class
    # sfm.show_rendered_images()
    


    # From dense depth prediction of a neural network
    # if use_pcd_from_depth_prediction:
    #     positions, colors = init_dense_pcd_from_network(viewpoint_stack, reconstruction, num_points = 50000)
    #     sfm.add_dense_point_cloud(positions=positions, colors=colors)
    

    # sfm.start_calib_iter = 250
    # sfm.stop_calib_iter = 500

    # sfm.start_pose_iter = 200
    # sfm.stop_pose_iter = 500

    # sfm.start_gaussian_iter = 0
    # sfm.stop_gaussian_iter = 100000

    # sfm.add_dense_pcd_iter = 500


    # sfm.require_calibration = True
    # sfm.allow_lens_distortion = True
    

    # sfm_process = mp.Process(target=sfm.optimize)
    # sfm_process.start()

  
    # torch.cuda.synchronize()

    # if use_gui:
    #     q_main2vis.put(gui_utils.GaussianPacket(finish=True))
    #     gui_process.join()
    #     sfm_gui.Log("GUI Stopped and joined the main thread", tag="GUI")
    

    # Fig = Viewer(viewpoint_stack=sfm.viewpoint_stack,  gaussians_gl= create_gaussians_gl(sfm.gaussians))



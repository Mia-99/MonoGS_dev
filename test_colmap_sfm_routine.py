
import sys, os

import torch
import torch.multiprocessing as mp

import numpy as np
import pathlib

import pycolmap
from gaussian_splatting.utils.graphics_utils import BasicPointCloud

from sfm import SFM
from sfm import DepthAnything

from gaussian_splatting.scene.gaussian_model_GS import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gui import gui_utils, sfm_gui

import time
from argparse import ArgumentParser, Namespace
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams

from gaussian_splatting.utils.general_utils import safe_state
from utils.multiprocessing_utils import FakeQueue, clone_obj

from PIL import Image
from gaussian_splatting.utils.general_utils import PILtoTorch

import open3d as o3d

# pip install pycolmap
class ColMap:

    def __init__ (self, image_dir = None):

        self.reconstruction = None

        if image_dir is not None:
            self.run(image_dir)


    def run(self, image_dir=None):

        self.image_dir = image_dir

        if image_dir is None:
            return
        image_dir = pathlib.Path(image_dir)

        output_path =  image_dir.parent / "sparse"
        output_path.mkdir(parents="False", exist_ok="True")
        database_path = output_path / "database.db"

        pycolmap.extract_features(database_path, image_dir)
        pycolmap.match_exhaustive(database_path)
        maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)

        self.reconstruction = maps[0]

        self.reconstruction.write(output_path )
        # self.reconstruction.write_text(output_path )  # text format
        self.reconstruction.export_PLY(output_path / "points3D.ply")  # PLY format
        print(self.reconstruction.summary())


    def getCameras(self, downsample_scale = 1.0):
        camera_stack = []
        camera_centers = []
        calib_stack, focal0, kappa0 = self.getCalibration()
        posed_img_stack = self.getCamPosedImages()
        for idx, item in posed_img_stack.items():
            R, T, imgname = item
            # K, kappa = calib_stack[idx]
            image_path = os.path.join(self.image_dir, os.path.basename(imgname))
            image = Image.open(image_path)
            # adjust image resolution if necessary
            orig_w, orig_h = image.size
            imgsize = round(orig_w/(downsample_scale)), round(orig_h/(downsample_scale))

            resized_image_rgb = PILtoTorch(image, imgsize)
            gt_image = resized_image_rgb[:3, ...]

            image_height = gt_image.shape[1]
            image_width = gt_image.shape[2]
            
            fx = focal0 / downsample_scale
            fy = focal0 / downsample_scale
            cx = (image_width + 1) * 0.5
            cy = (image_height + 1) * 0.5
            kappa = kappa0 / downsample_scale
            # fx = K[0, 0] / downsample_scale
            # fy = K[1, 1] / downsample_scale
            # cx = K[0, 2] / downsample_scale
            # cy = K[1, 2] / downsample_scale
            # kappa = kappa / downsample_scale

            cam = Camera (
                        uid = idx,
                        color = gt_image,
                        depth = None,
                        image_height = image_height,
                        image_width = image_width,
                        R = R, T = T,
                        fx = fx,
                        fy = fy,
                        cx = cx,
                        cy = cy,
                        fovx = None,
                        fovy = None,
                        kappa = kappa,
                        trans=np.array([0.0, 0.0, 0.0]),
                        scale=1.0,
                        gt_alpha_mask = None,
                        device="cuda:0",
            )
            camera_stack.append(cam)
            camera_centers.append( - R.transpose() @ T.reshape((3, 1)) ) # camera center
        # getNerfppNorm copied from 3DGS original implementation
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal
        center, diagonal = get_center_and_diag(camera_centers)
        radius = diagonal * 1.1
        translate = -center
        return camera_stack, {"translate": translate, "radius": radius}



    def getPointCloud(self):
        positions = []
        colors = []
        normals = []
        for point3D_id, point3D in self.reconstruction.points3D.items():
            positions.append(point3D.xyz)
            colors.append(point3D.color / 255.0) # use normalzied colors, which willl be passed to SH
        positions = np.array(positions)
        colors = np.array(colors)
        return positions, colors


    # The reconstructed pose of an image is specified as 
    # the projection from world to the camera coordinate system of an image using
    # a quaternion (QW, QX, QY, QZ) and a translation vector (TX, TY, TZ).
    # The coordinates of the projection/camera center are given by -R^t * T
    # The local camera coordinate system of an image is defined in a way that:
    #   * the X axis points to the right,
    #   * the Y axis to the bottom,
    #   * the Z axis to the front as seen from the image.
    # Bring a world point X_world to camera frame
    # X_cam = R * X_world  +  t
    def getCamPosedImages(self):
        pose_stack = {}
        for image_id, image in self.reconstruction.images.items():
            pose = image.cam_from_world
            qvec = pose.rotation.quat
            tvec = pose.translation
            # [ R, T ] is a tranformation from world frame to camera frame
            R = self.qvec2rotmat( qvec )
            T = np.array( tvec )
            pose_stack[image_id] = (R, T, image.name)
        return pose_stack


    def getCalibration(self):
        calib_stack = {}
        f = 0.0
        k = 0.0
        for camera_id, camera in self.reconstruction.cameras.items():
            focal = camera.params[0]
            kappa = camera.params[3]
            cx = camera.params[1]
            cy = camera.params[2]
            K = np.array([[focal, 0.0, cx],
                          [0.0, focal, cy],
                          [0.0, 0.0,  1.0]])
            calib_stack[camera_id] = (K, kappa)
            f += focal
            k += kappa
        return calib_stack,  f/len(calib_stack),  k/len(calib_stack)


    @staticmethod
    # copied from 3DGS colmap.loader.py
    def qvec2rotmat(qvec):
        return np.array([
            [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,   2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],  2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],   1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,  2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],   2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],  1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])







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


    opt.iterations = 1000
    opt.densification_interval = 50
    opt.opacity_reset_interval = 350
    opt.densify_from_iter = 49
    opt.densify_until_iter = 700
    opt.densify_grad_threshold = 0.0002





    use_colmap_point_cloud = True


    image_dir = "/home/fang/SURGAR/Colmap_Test/Fountain/images"
    # image_dir = "/home/fang/SURGAR/Colmap_Test/truck/images"


    # perform colmap reconstruction
    reconstruction = ColMap(image_dir)

    # extract reconstruction information: 1. posedCameras, 2. 3Dpointcloud
    viewpoint_stack, scale_info = reconstruction.getCameras()
    positions, colors = reconstruction.getPointCloud()    
    pcd = BasicPointCloud(points=positions, colors=colors, normals=None)

    # initialize 3D Gaussians
    print(f"scale_info = {scale_info}")
    cameras_extent = scale_info["radius"]
    gaussians = GaussianModel(sh_degree=0)


    
    if use_colmap_point_cloud:
        gaussians.create_from_pcd(pcd, cameras_extent)

    else:
        cam = viewpoint_stack[0]

        rgb_raw = (cam.original_image *255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        # use depth prediction from a Neural network
        depth_raw = DepthAnything().eval(rgb_raw)

        rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
        depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        gaussians.create_from_image_and_depth(cam, rgb, depth, downsample_factor = 8, point_size = 0.01)




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
        time.sleep(3)


    print(f"Run with image W: { viewpoint_stack[0].image_width },  H: { viewpoint_stack[0].image_height }")

    sfm = SFM(pipe, q_main2vis, q_vis2main, use_gui, viewpoint_stack, gaussians, opt, cameras_extent)
    sfm.add_calib_noise_iter = -1
    sfm.start_calib_iter = 50
    sfm.require_calibration = True
    sfm.allow_lens_distortion = True
    

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



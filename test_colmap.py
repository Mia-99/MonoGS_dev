
import sys, os

import torch
import torch.multiprocessing as mp

import numpy as np
import pathlib

import pycolmap
from gaussian_splatting.utils.graphics_utils import BasicPointCloud

from sfm import SFM
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gui import gui_utils, sfm_gui

import time
from argparse import ArgumentParser, Namespace
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams

from gaussian_splatting.utils.general_utils import safe_state
from utils.multiprocessing_utils import FakeQueue, clone_obj

from PIL import Image
from gaussian_splatting.utils.general_utils import PILtoTorch



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

        output_path =  image_dir.parent
        database_path = output_path / "database.db"

        pycolmap.extract_features(database_path, image_dir)
        pycolmap.match_exhaustive(database_path)
        maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)

        self.reconstruction = maps[0]

        self.reconstruction.write(output_path)
        # self.reconstruction.write_text(output_path )  # text format
        # self.reconstruction.export_PLY(output_path / "rec.ply")  # PLY format
        print(self.reconstruction.summary())


    def getCameras(self):
        camera_stack = []
        calib_stack = self.getCalibration()
        posed_img_stack = self.getCamPosedImages()
        for idx, item in posed_img_stack.items():
            R, T, imgname = item
            image_path = os.path.join(self.image_dir, os.path.basename(imgname))
            image = Image.open(image_path)

            orig_w, orig_h = image.size

            resolution_scale = 1.0
            resolution = 4.0

            imgsize = round(orig_w/(resolution_scale * resolution)), round(orig_h/(resolution_scale * resolution))

            resized_image_rgb = PILtoTorch(image, imgsize)
            gt_image = resized_image_rgb[:3, ...]

            image_height = gt_image.shape[1]
            image_width = gt_image.shape[2]

            K, kappa = calib_stack[idx]
            fx = K[0, 0] / resolution
            fy = K[1, 1] / resolution
            cx = K[0, 2] / resolution
            cy = K[1, 2] / resolution
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
        return camera_stack


    # def getNerfppNorm(self):
    #     def get_center_and_diag(cam_centers):
    #         cam_centers = np.hstack(cam_centers)
    #         avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    #         center = avg_cam_center
    #         dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    #         diagonal = np.max(dist)
    #         return center.flatten(), diagonal

    #     cam_centers = []

    #     for cam in cam_info:
    #         W2C = getWorld2View2_GS(cam.R, cam.T)
    #         C2W = np.linalg.inv(W2C)
    #         cam_centers.append(C2W[:3, 3:4])

    #     center, diagonal = get_center_and_diag(cam_centers)
    #     radius = diagonal * 1.1

    #     translate = -center

    #     return {"translate": translate, "radius": radius}


    def getPointCloud(self):
        positions = []
        colors = []
        normals = []
        for point3D_id, point3D in self.reconstruction.points3D.items():
            positions.append(point3D.xyz)
            colors.append(point3D.color)
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
        for camera_id, camera in self.reconstruction.cameras.items():
            focal = camera.params[0]
            kappa = camera.params[3]
            cx = camera.params[1]
            cy = camera.params[2]
            K = np.array([[focal, 0.0, cx],
                          [0.0, focal, cy],
                          [0.0, 0.0,  1.0]])
            calib_stack[camera_id] = (K, kappa)
        return calib_stack


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


    opt.iterations = 500
    opt.densification_interval = 50
    opt.opacity_reset_interval = 350
    opt.densify_from_iter = 49
    opt.densify_until_iter = 3000
    opt.densify_grad_threshold = 0.0002







    image_dir = "/home/fang/SURGAR/Colmap_Test/Fountain/images"

    reconstruction = ColMap(image_dir)
    viewpoint_stack = reconstruction.getCameras()
    positions, colors = reconstruction.getPointCloud()    

    pcd = BasicPointCloud(points=positions, colors=colors, normals=None)

    cameras_extent = 0.1
    gaussians = GaussianModel(0)
    gaussians.create_from_pcd(pcd, cameras_extent)




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

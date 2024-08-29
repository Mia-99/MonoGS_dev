
import os
import numpy as np
import pathlib
import pycolmap

from PIL import Image


from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.utils.general_utils import PILtoTorch




# pip install pycolmap
class ColMap:

    def __init__ (self, image_dir = None):

        self.reconstruction = None

        self.image_dir = image_dir

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






# a function to create a list of Camera classes in 3DGS/MonoGS
def assemble_3DGS_cameras(colmap : ColMap, downsample_scale = 1.0):
    camera_stack = []
    camera_centers = []
    calib_stack, focal0, kappa0 = colmap.getCalibration()
    posed_img_stack = colmap.getCamPosedImages()
    for idx, item in posed_img_stack.items():
        R, T, imgname = item
        # K, kappa = calib_stack[idx]
        image_path = os.path.join(colmap.image_dir, os.path.basename(imgname))
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




if __name__ == "__main__":

    image_dir = "/home/fang/SURGAR/Colmap_Test/Fountain/images"

    # perform colmap reconstruction
    reconstruction = ColMap(image_dir)

    # extract reconstruction information: 1. posedCameras, 2. 3Dpointcloud.  3. Calibrations
    positions, colors = reconstruction.getPointCloud()
    posed_img_stack = reconstruction.getCamPosedImages()
    calib_stack, focal0, kappa0 = reconstruction.getCalibration()

    # interface to 3DGS
    pcd = BasicPointCloud(points=positions, colors=colors, normals=None)
    viewpoint_stack, scale_info = assemble_3DGS_cameras(reconstruction)
 
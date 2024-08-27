

import numpy as np
import pathlib

import pycolmap





# pip install pycolmap
class ColMap:

    def __init__ (self, image_dir = None):
        self.reconstruction = None
        if image_dir is not None:
            self.run(image_dir)


    def run(self, image_dir=None):
        
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
    def getCamPoses(self):
        pose_stack = {}
        for image_id, image in self.reconstruction.images.items():
            pose = image.cam_from_world
            qvec = pose.rotation.quat
            tvec = pose.translation
            # [ R, T ] is a tranformation from world frame to camera frame
            R = self.qvec2rotmat( qvec )
            T = np.array( tvec )
            pose_stack[image_id] = (R, T)

        

    def getPointCloud(self):
        for point3D_id, point3D in self.reconstruction.points3D.items():
            print(point3D_id, point3D)
            print(point3D.xyz)



    def getCalibration(self):
        camera_stack = {}
        for camera_id, camera in self.reconstruction.cameras.items():
            focal = camera.params[0]
            kappa = camera.params[3]
            cx = camera.params[1]
            cy = camera.params[2]
            K = np.array([[focal, 0.0, cx],
                          [0.0, focal, cy],
                          [0.0, 0.0,  1.0]])
            camera_stack[camera_id] = (K, kappa)
        return camera_stack


    @staticmethod
    # copied from 3DGS colmap.loader.py
    def qvec2rotmat(qvec):
        return np.array([
            [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,   2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],  2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],   1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,  2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],   2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],  1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])






if __name__ == "__main__":

    image_dir = "/home/fang/SURGAR/Colmap_Test/Fountain/images"

    reconstruction = ColMap(image_dir)

    # print( reconstruction.getCalibration() )
    # print (reconstruction.getPointCloud())
    print( reconstruction.getCamPoses())



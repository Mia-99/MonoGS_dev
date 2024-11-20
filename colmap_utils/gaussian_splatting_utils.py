import os
import numpy as np
import open3d as o3d
from colmap_utils.colmap import ColMap

from PIL import Image

from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.utils.general_utils import PILtoTorch


# a function to create a list of Camera classes in 3DGS/MonoGS
def assemble_3DGS_cameras(colmap : ColMap, downsample_scale = 1.0,  use_same_calib = True):
    camera_stack = []
    camera_centers = []
    posed_image_stack = colmap.getCamPosedImages()

    for image_id, item in posed_image_stack.items():
        R, T, imgname, K, kappa = item
        
        image_path = os.path.join(colmap.image_dir, os.path.basename(imgname))
        image = Image.open(image_path)
        # adjust image resolution if necessary
        orig_w, orig_h = image.size
        imgsize = round(orig_w/(downsample_scale)), round(orig_h/(downsample_scale))

        resized_image_rgb = PILtoTorch(image, imgsize)
        gt_image = resized_image_rgb[:3, ...]

        image_height = gt_image.shape[1]
        image_width = gt_image.shape[2]
        
        fx = K[0, 0]  / downsample_scale
        fy = K[1, 1]  / downsample_scale
        cx = K[0, 2]  / downsample_scale
        cy = K[1, 2]  / downsample_scale
        kappa = kappa / downsample_scale

        cam = Camera (
                    uid = image_id,
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




def create_trajectory_lineset(viewpoint_stack, color=[0, 0, 1]):
    camera_centers = []
    for viewpoint in viewpoint_stack:
        camera_centers.append ( viewpoint.camera_center.detach().cpu().numpy() )
    points = np.array( camera_centers )

    lines = []
    for i in range(len(camera_centers)-1):
        lines.append( [i, i+1] )

    colors = [color for i in range(len(lines))]

    odometry_line_set = o3d.geometry.LineSet()
    odometry_line_set.points = o3d.utility.Vector3dVector(points)
    odometry_line_set.lines = o3d.utility.Vector2iVector(lines)
    odometry_line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return odometry_line_set





if __name__ == "__main__":

    image_dir = "/home/fang/SURGAR/Colmap_Test/Fountain/images"

    # perform colmap reconstruction
    reconstruction = ColMap(image_dir)



    # extract reconstruction information: 1. posedCameras, 2. 3Dpointcloud.  3. Calibrations
    positions, colors = reconstruction.getPointCloud()
    posed_img_stack = reconstruction.getCamPosedImages()

    # interface to 3DGS
    pcd = BasicPointCloud(points=positions, colors=colors, normals=None)
    viewpoint_stack, scale_info = assemble_3DGS_cameras(reconstruction)

    sparse_depth_stack = reconstruction.getSparseDepthFromImage(image_id = 1)


    try:

        # Create a visualizer
        WIDTH = 1280
        HEIGHT = 720

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=WIDTH, height=HEIGHT)


        # add poinit-cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        # opt.point_show_normal = True
        
        
        # add cameras
        for image_id, item in posed_img_stack.items():
            R, T, imgname, K, kappa = item
            intrinsic = K            
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = T
            cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT, intrinsic=intrinsic, extrinsic=extrinsic)
            vis.add_geometry(cameraLines)

        odometryLines = create_trajectory_lineset(viewpoint_stack)
        vis.add_geometry(odometryLines)


        # visualize and block
        vis.run()
    
    except:
        pass
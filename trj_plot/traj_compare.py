

import json
import numpy as np
import torch
import os
import evo
import evo.main_config
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
import copy
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def load_pose(file_path):
    poses = read_json_file(file_path)
    if poses is None:
        return False
    trj_id = poses['trj_id']
    np_trj_est = np.array(poses['trj_est'])
    np_trj_gt = np.array(poses['trj_gt'])
    print(np_trj_est.shape)
    assert len(trj_id) == len(np_trj_est) == len(np_trj_gt)
    # list to tensor
    trj_est = []
    trj_gt = []
    for i in range(len(trj_id)):
        T = np_trj_est[i, :, :]
        # cam_pose_est = np.linalg.inv(T)
        cam_pose_est = T
        # trj_est.append(torch.tensor(cam_pose_est, device="cuda"))
        trj_est.append(cam_pose_est)
        T = np_trj_gt[i, :, :]
        # cam_pose_gt = np.linalg.inv(T)
        cam_pose_gt = T
        trj_gt.append(cam_pose_gt)
        # trj_gt.append(torch.tensor(cam_pose_gt, device="cuda"))
        
    return trj_gt, trj_est
    
base_traj_path = "./trj_0661.json"
cali_traj_path = "./trj_final.json"

trj_cali_gt , trj_est_cali = load_pose(cali_traj_path)
trj_gsslam_gt , trj_est_gsslam = load_pose(base_traj_path)

traj_ref = PosePath3D(poses_se3=trj_cali_gt)
traj_ref_gsslam = PosePath3D(poses_se3=trj_gsslam_gt)
traj_est_cali = PosePath3D(poses_se3=trj_est_cali)
traj_est_gsslam = PosePath3D(poses_se3=trj_est_gsslam)

traj_est_aligned_cali = copy.deepcopy(traj_est_cali)
traj_est_aligned_gsslam = copy.deepcopy(traj_est_gsslam)
traj_est_aligned_cali.align(traj_ref, correct_scale=True)
traj_est_aligned_gsslam.align(traj_ref_gsslam, correct_scale=True)

## RMSE
pose_relation = metrics.PoseRelation.translation_part
data1 = (traj_ref, traj_est_aligned_cali)
ape_metric1 = metrics.APE(pose_relation)
ape_metric1.process_data(data1)
# ape_stat1 = ape_metric1.get_statistic(metrics.StatisticsType.rmse)
ape_stats1 = ape_metric1.get_all_statistics()
plot_dir = "./"

pose_relation = metrics.PoseRelation.translation_part
data2 = (traj_ref_gsslam, traj_est_aligned_gsslam)
ape_metric2 = metrics.APE(pose_relation)
ape_metric2.process_data(data2)
ape_stats2 = ape_metric2.get_all_statistics()

xyz_ref = traj_ref_gsslam.positions_xyz
qut_ref = traj_ref_gsslam.orientations_quat_wxyz
qut_ref_xyzw = qut_ref[:, [1, 2, 3, 0]]


xyz_gsslam = traj_est_aligned_gsslam.positions_xyz
qut_gsslam = traj_est_aligned_gsslam.orientations_quat_wxyz
qut_gsslam_xyzw = qut_gsslam[:, [1, 2, 3, 0]]

xyz_calib = traj_est_aligned_cali.positions_xyz
qut_calib = traj_est_aligned_cali.orientations_quat_wxyz
qut_calib_xyzw = qut_calib[:, [1, 2, 3, 0]]
# print("xyz_ref: ", xyz_ref)

mesh_file_path_name = '/datasets/office1_mesh.ply'


# read mesh
mesh = o3d.io.read_triangle_mesh(mesh_file_path_name)
if not mesh.has_vertex_normals():
  mesh.compute_vertex_normals()

# mesh.textures  = ?
vertex_positions = np.asarray(mesh.vertices)
vertex_normals = np.asarray(mesh.vertex_normals)
vertex_colors = np.asarray(mesh.vertex_colors)
triangle_indices = np.asarray(mesh.triangles)

rr.init("rerun_example_my_data", spawn=True)

rr.log(
    "strips",
    rr.LineStrips3D(
        [
            xyz_ref.tolist(),
            xyz_gsslam.tolist(),
            xyz_calib.tolist(),
        ],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        radii=[0.01, 0.01, 0.01],
        labels=["one strip here", "and one strip there", "and one strip there"],
    ),
)

# rng = np.random.default_rng(12345)
# image = rng.uniform(0, 255, size=[3, 3, 3])

# give me intrinsics
intrinsics = np.array([
    [500.0, 0, 399.5],
    [0, 500.0, 299.5],
    [0, 0, 1]
])

interval = 2
for i in range(len(xyz_ref)):
    # rr.log(
    #     "world/view2", rr.Transform3D(translation=[0, 0, 20],
    #     rotation=RotationAxisAngle(axis=[0, 1, 0], angle=Angle(rad=2*pi / 4)),
    #     scale=1,
    #     from_parent=True) # from_parent=True means camera transform is "camera from world"
    # )
    # rr.log("world/view2", rr.ViewCoordinates.RDF, static=True) # X=Right, Y=Down, Z=Forward
    # # Log camera intrinsics
    # rr.log("world/view2/image", rr.Pinhole(focal_length=5, width=20, height=20))
    
    if i % interval != 0:
        continue
    rr.log(
        f"world/camera{i}",
        rr.Transform3D(translation=xyz_ref[i], scale=1, rotation=rr.Quaternion(xyzw=qut_ref_xyzw[i])),
    )
    rr.log(f"world/camera{i}", rr.Pinhole(
        resolution=[800, 600],
        image_from_camera=intrinsics, 
        camera_xyz = rr.ViewCoordinates.RDF,
    ))

# rr.log("world/image", rr.Image(image))

rr.log(
    "ref",
    rr.Points3D(
        xyz_ref.tolist(),
        colors=[255, 0, 0],
        radii=0.01,
    ),
)
rr.log(
    "GSSLAM",
    rr.Points3D(
        xyz_gsslam.tolist(),
        colors=[0, 255, 0],
        radii=0.01,
    ),
)
rr.log(
    "Ours",
    rr.Points3D(
        xyz_calib.tolist(),
        colors=[0, 0, 255],
        radii=0.01,
    ),
)

# rr.log(
#     "triangle",
#     rr.Mesh3D(
#         vertex_positions=vertex_positions,
#         vertex_normals=vertex_normals,
#         vertex_colors=vertex_colors,
#         triangle_indices=triangle_indices,
#     ),
# )





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


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def load_pose(file_path):
    poses = read_json_file(os.path.join(file_path, 'plot', 'trj_final.json'))
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
        cam_pose_est = np.linalg.inv(T)
        # trj_est.append(torch.tensor(cam_pose_est, device="cuda"))
        trj_est.append(cam_pose_est)
        T = np_trj_gt[i, :, :]
        cam_pose_gt = np.linalg.inv(T)
        trj_gt.append(cam_pose_gt)
        # trj_gt.append(torch.tensor(cam_pose_gt, device="cuda"))
        
    return trj_gt, trj_est
    
base_traj_path = "/workspaces/src/MonoGS_dev/results/monocular/replica_small/office0/2024-11-07-22-33-23"
cali_traj_path = "/workspaces/src/MonoGS_dev/results/monocular/replica_small_cali/office0_v6/2024-11-07-17-50-10"

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

plot_mode = evo.tools.plot.PlotMode.xy
fig = plt.figure()
ax = evo.tools.plot.prepare_axis(fig, plot_mode)
ax.set_title(f"Comparison of trajectories on the Relica Office0 dataset")
evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
evo.tools.plot.traj(ax, plot_mode, traj_est_aligned_gsslam, "-", "red", "gt_gsslam")
evo.tools.plot.traj(ax, plot_mode, traj_est_aligned_cali, "-", "blue", "cali")
# evo.tools.plot.traj_colormap(
#     ax,
#     traj_est_aligned_cali,
#     ape_metric1.error,
#     plot_mode,
#     min_map=ape_stats1["min"],
#     max_map=ape_stats1["max"],
# )
# evo.tools.plot.traj_colormap(
#     ax,
#     traj_est_aligned_gsslam,
#     ape_metric2.error,
#     plot_mode,
#     min_map=ape_stats2["min"],
#     max_map=ape_stats2["max"],
# )
ax.legend()
plt.savefig(os.path.join(plot_dir, "evo_2dplot.pdf"), dpi=90)

# # if label == "final":
plot_mode = evo.tools.plot.PlotMode.xyz
fig = plt.figure()
ax = evo.tools.plot.prepare_axis(fig, plot_mode)
ax.set_title(f"Comparison of trajectories on the Relica Office0 dataset")
# SETTINGS.plot_axis_marker_scale = 0.1
# SETTINGS.plot_reference_axis_marker_scale = 0.1
# SETTINGS.plot_pose_correspondences = True
evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
evo.tools.plot.traj(ax, plot_mode, traj_est_aligned_gsslam, "-", "red", "gt_gsslam")
evo.tools.plot.traj(ax, plot_mode, traj_est_aligned_cali, "-", "blue", "cali")
ax.legend()
plt.savefig(os.path.join(plot_dir, "evo_3dplot.pdf"), dpi=90)
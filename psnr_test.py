# load gaussians
# load dataset and cameras

# compute psnr, ssim, lpips

import pickle

import numpy as np
import os

import torch

from gui.gl_render import util as util
from gui.gl_render import util_gau_lima as util_gau
from gui.gl_render import render_ogl as render_ogl
from utils.camera_utils import Camera
from utils.config_utils import load_config
from utils_cali.dataset_cali import load_dataset
from munch import munchify

from utils_cali.eval_cali_utils import eval_rendering

def gaussian_model_to_gaussian_data(gaussian_model):
    xyz = gaussian_model.get_xyz.cpu().numpy() 
    opacity = gaussian_model.get_opacity.cpu().numpy()
    scale = gaussian_model.get_scaling.cpu().numpy()
    rot = gaussian_model.get_rotation.cpu().numpy()
    features_dc = gaussian_model.get_features.cpu().numpy() #output with features_cd and features_rest
    
    # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    # assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    # for idx, attr_name in enumerate(extra_f_names):
    #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    # features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    # features_extra = np.transpose(features_extra, [0, 2, 1])
    extra_f_names = []
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                    features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)
    # sh = gaussian_model.get_features.detach().cpu().numpy()[:, 0, :]
    sh = gaussian_model.max_sh_degree
    # exit()  
    return util_gau.GaussianData(xyz, rot, scale, opacity, shs)

replica_config_path = "./configs/mono/replica_small/office3_sp.yaml"
# replica_cali_config_path = "./configs/mono/replica_small_cali/office3_v5_sp.yaml"


replica_config = load_config(replica_config_path)
model_params_origin = munchify(replica_config["model_params"])
pipeline_params = munchify(replica_config["pipeline_params"])
replica_origin_dataset = load_dataset(model_params_origin, model_params_origin.source_path, config=replica_config)

# replica_cali_config = load_config(replica_cali_config_path)
# model_params = munchify(replica_cali_config["model_params"])
# replica_cali_dataset = load_dataset(model_params, model_params.source_path, config=replica_cali_config)
                                    
gaussians_path = "/workspaces/src/MonoGS_dev/results/monocular/replica_small_cali/office0_v6/2024-11-04-22-05-14/gs/instance.pkl"
with open(gaussians_path, "rb") as f:
    gaussians = pickle.load(f)
# print(gaussians)
# gaussians_gl = gaussian_model_to_gaussian_data(gaussians)
bg_color = [0, 0, 0]
background=torch.tensor(bg_color, dtype=torch.float32, device="cuda")
path = "./results/monocular/replica_small_cali/office0_v6/2024-11-04-22-05-14/test/"

kf_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
frames = []
for i in range(0, 600):
    viewpoint_origin = Camera.init_from_dataset(replica_origin_dataset, i)
    viewpoint_origin.compute_grad_mask(replica_config)
    frames.append(viewpoint_origin)
eval_rendering(frames, gaussians, replica_origin_dataset, path, pipeline_params, background=background, kf_indices = kf_ids, iteration='before_opt')
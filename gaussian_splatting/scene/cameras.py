#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, focal2fov, getWorld2View2_GS

from utils.pose_utils import SO3_exp


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name



        self.R = torch.from_numpy(R)
        self.T = torch.from_numpy(T)


        # print(f"self.R.shape = {self.R.shape},  type = {self.R.dtype}")
        # print(f"self.T.shape = {self.T.shape},  type = {self.T.dtype}")

        # add some noise and see if the algorithm can correctly optimize it
        noiseDeltaR = SO3_exp(torch.tensor([0.01, 0.01, 0.01])).double() 
        noiseDeltaT = torch.tensor([0.1, 0.1, 0.1], dtype=torch.double)

        self.T = self.T + noiseDeltaT
        self.R = self.R @ noiseDeltaR




        # T = torch.eye(4, device=data_device)
        # self.R = T[:3, :3]
        # self.T = T[:3, 3]
        self.R_gt = R
        self.T_gt = T


        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale


        self.device = data_device

        self.depth = None
        self.grad_mask = None


        self.fx = fov2focal(self.FoVx, self.image_width)
        self.fy = fov2focal(self.FoVy, self.image_height)

        # approximate for now
        self.cx = (self.image_width - 1) * 0.5
        self.cy = (self.image_height - 1) * 0.5


        # add self-calibration variables
        self.focal = 100
        self.kappa = 1.0


        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=data_device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=data_device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=data_device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=data_device)
        )



    @property
    def projection_matrix(self):
        # return getProjectionMatrix2(
        #             znear=self.znear,
        #             zfar=self.zfar,
        #             fx=self.fx,
        #             fy=self.fy,
        #             cx=self.cx,
        #             cy=self.cy,
        #             W=self.image_width,
        #             H=self.image_height,
        #         ).transpose(0, 1)
        return getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(device=self.device)



    # @staticmethod
    # def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
    #     projection_matrix = getProjectionMatrix2(
    #         znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
    #     ).transpose(0, 1)
    #     return Camera(
    #         uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
    #     )

    # @property
    # def world_view_transform(self):
    #     return getWorld2View2(self.R, self.T).transpose(0, 1)


    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        R = T[:3, :3].cpu().numpy()
        t = T[:3, 3].cpu().numpy()
        # print(f"R = {R}")
        # print(f"t = {t}")
        img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

        # print(f"Camera:")
        # print(f"      uid = {uid}")
        # print(f"     FoVx = {FoVx}")
        # print(f"     FoVy = {FoVy}")
        # print(f"        H = {H}")
        # print(f"        W = {W}")
        # print(f"        R = {R}")
        # print(f"        t = {t}")

        return Camera(
            uid, R, t, FoVx, FoVy, img, None, None, uid,
        )
    

    @property
    def world_view_transform(self):
        # print(f"self.R = {self.R},   self.T = {self.T}")
        T = getWorld2View2(self.R, self.T, torch.from_numpy(self.trans), self.scale).transpose(0, 1).to(device=self.device)
        # print(f"world_view_transform = {T}")
        return T


    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)


    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]


    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)





class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


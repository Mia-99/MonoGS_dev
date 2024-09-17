
import torch
import torch.optim.lr_scheduler as lr_scheduler

from utils.pose_utils import update_pose

import numpy as np

from numpy.polynomial import Polynomial, Chebyshev

import matplotlib.pyplot as plt


class LineDetection:

    def __init__(self, xdata, ydata, deg = 7) -> None:

        self._Chebyshev_poly = Chebyshev.fit(xdata, ydata, deg=deg)

        self.poly = self._Chebyshev_poly.convert(kind=Polynomial, domain=self._Chebyshev_poly.domain,  window=self._Chebyshev_poly.window)

        self.poly_deriv = self.poly.deriv(1)

        self.slopes = self.poly_deriv(xdata)


        plt.plot(xdata, ydata, 'o')

        xx, yy = self._Chebyshev_poly.linspace()
        plt.plot(xx, yy, '+')

        xxc, yyc = self.poly.linspace()
        plt.plot(xxc, yyc, lw=2)





        plt.title(r"$L(f) = af^2+bf+c  \Leftrightarrow \nabla L(f) = 2 a f + b $")
        plt.xlabel(r"focal")
        plt.ylabel(r"$\nabla L(f)$")
        plt.show()


        xxd, yyd = self.poly_deriv.linspace()
        plt.plot(xxd, yyd, lw=5)
        plt.plot(xdata, self.slopes, '*')
        plt.show()


        print(self.poly.coef)
        print(self.poly.domain)
        print(self.poly.window)
        print(self.poly)
        print(self.slopes)        
        print(self.poly_deriv)
        





    def extractline (self):
        pass





class CalibrationOptimizer:


    def __init__(self, viewpoint_stack) -> None:

        self.viewpoint_stack = viewpoint_stack

        self.calibration_groups = {}
        self.focal_delta_groups = {}
        self.kappa_delta_groups = {}

        self.focal_optimizer = None
        self.kappa_optimizer = None

        self.__init_calibration_groups()
        self.__init_optimizers()
        self.zero_grad()


        self.focal_grad_stack = []
        self.focal_stack = []



    def __init_calibration_groups(self):
        self.calibration_groups = {}
        for viewpoint_cam in self.viewpoint_stack:
            calib_id = viewpoint_cam.calibration_identifier
            if calib_id not in self.calibration_groups:
                self.calibration_groups[ calib_id ] = []
            self.calibration_groups[ calib_id ].append(viewpoint_cam)
        for calib_id, cam_stack in self.calibration_groups.items():
            self.focal_delta_groups [ calib_id ] = torch.tensor([0.0], requires_grad=True, device=cam_stack[0].device)
            self.kappa_delta_groups [ calib_id ] = torch.tensor([0.0], requires_grad=True, device=cam_stack[0].device)
            self.focal_delta_groups [ calib_id ].grad = torch.tensor([0.0], device=cam_stack[0].device)
            self.kappa_delta_groups [ calib_id ].grad = torch.tensor([0.0], device=cam_stack[0].device)



    def __init_optimizers(self):
        focal_opt_params = []
        kappa_opt_params = []
        for calib_id, cam_stack in self.calibration_groups.items():
            focal_opt_params.append(
                    {
                        "params": [ self.focal_delta_groups [ calib_id ] ],
                        "lr": 0.1,
                        "name": "calibration_f_{}".format(calib_id),
                    }
                )
            kappa_opt_params.append(
                    {
                        "params": [ self.kappa_delta_groups [ calib_id ] ],
                        "lr": 0.001,
                        "name": "calibration_k_{}".format(calib_id),
                    }
                )
        self.focal_optimizer = torch.optim.NAdam(focal_opt_params)
        self.kappa_optimizer = torch.optim.NAdam(kappa_opt_params)
        
        



    # put it under .grad? to be used with optimizers
    def __update_focal_gradients (self):
        for calib_id, cam_stack in self.calibration_groups.items():

            self.focal_delta_groups [ calib_id ].data.fill_(0)
            self.focal_delta_groups [ calib_id ].grad.zero_()

            for viewpoint_cam in cam_stack:
                self.focal_delta_groups [ calib_id ].grad += viewpoint_cam.cam_focal_delta.grad



    # put it under .grad? to be used with optimizers
    def __update_kappa_gradients (self):
        for calib_id, cam_stack in self.calibration_groups.items():

            self.kappa_delta_groups [ calib_id ].data.fill_(0)
            self.kappa_delta_groups [ calib_id ].grad.zero_()

            for viewpoint_cam in cam_stack:
                self.kappa_delta_groups [ calib_id ].grad += viewpoint_cam.cam_kappa_delta.grad



    def focal_step(self, loss=None):
        self.__update_focal_gradients()

        focal_grad_vec = []
        focal_vec = []

        # L-BFGS closure
        def closure():
            return loss
        
        if type(self.focal_optimizer).__name__ == 'LBFGS':
            self.focal_optimizer.step(closure) # to use LBFGS
        else:
            self.focal_optimizer.step()

        for calib_id, cam_stack in self.calibration_groups.items():
            focal_delta = self.focal_delta_groups [ calib_id ].data.cpu().numpy()[0]
            for viewpoint_cam in cam_stack:
                focal = viewpoint_cam.fx
                viewpoint_cam.fx += focal_delta
                viewpoint_cam.fy += viewpoint_cam.aspect_ratio * focal_delta
            focal_grad  = self.focal_delta_groups [ calib_id ].grad.cpu().numpy()[0]
            print(f"\n\tfocal_update = {focal_delta},\tgradient = {focal_grad}")
                        
            focal_grad_vec.append(focal_grad)
            focal_vec.append(focal)

        self.focal_grad_stack.append(np.array(focal_grad_vec))
        self.focal_stack.append(np.array(focal_vec))

        # if np.linalg.norm( np.array(focal_grad_vec) ) < 0.00001:
        #     self.update_focal_learning_rate(lr=0.001)
        # if np.linalg.norm( np.array(focal_grad_vec) ) < 0.000001:
        #     self.update_focal_learning_rate(lr=0.0001)
        # if np.linalg.norm( np.array(focal_grad_vec) ) < 0.0000001:
        #     self.update_focal_learning_rate(lr=0.00001)


    def kappa_step(self):
        self.__update_kappa_gradients()
        self.kappa_optimizer.step()
        for calib_id, cam_stack in self.calibration_groups.items():
            kappa_delta = self.kappa_delta_groups [ calib_id ].data.cpu().numpy()[0]
            for viewpoint_cam in cam_stack:
                viewpoint_cam.kappa += kappa_delta
            kappa_grad  = self.kappa_delta_groups [ calib_id ].grad.cpu().numpy()[0]
            print(f"\tkappa_update = {kappa_delta},\tgradient = {kappa_grad}")



    def zero_grad(self):
        for viewpoint_cam in self.viewpoint_stack:
            viewpoint_cam.cam_focal_delta.data.fill_(0)
            viewpoint_cam.cam_kappa_delta.data.fill_(0)
            if viewpoint_cam.cam_focal_delta.grad is not None:
                viewpoint_cam.cam_focal_delta.grad.detach_()
                viewpoint_cam.cam_focal_delta.grad.zero_()
            if viewpoint_cam.cam_kappa_delta.grad is not None:
                viewpoint_cam.cam_kappa_delta.grad.detach_()
                viewpoint_cam.cam_kappa_delta.grad.zero_()



    def update_focal_learning_rate (self, lr = None, scale = None):
        for param_group in self.focal_optimizer.param_groups:
            if lr is not None:
                param_group["lr"] = lr
            if scale is not None:
                lr = param_group["lr"]
                param_group["lr"] = scale * lr if lr > 0.0001 else lr
        print(f"\nfocal_optimizer.param_groups:\n\t{self.focal_optimizer.param_groups}\n")



    def update_kappa_learning_rate (self, lr = None, scale = None):
        for param_group in self.kappa_optimizer.param_groups:
            if lr is not None:
                param_group["lr"] = lr
            if scale is not None:
                lr = param_group["lr"]
                param_group["lr"] = scale * lr if lr > 0.00001 else lr
        print(f"\nkappa_optimizer.param_groups:\n\t{self.kappa_optimizer.param_groups}\n")



    def get_focal_statistics (self):
        focal_grad_stack  = np.array(self.focal_grad_stack).transpose()
        focal_stack = np.array(self.focal_stack).transpose()
        return focal_stack, focal_grad_stack




class PoseOptimizer:

    def __init__(self, viewpoint_stack) -> None:

        self.viewpoint_stack = viewpoint_stack

        self.pose_optimizer = None

        self.__init_optimizer()
        self.zero_grad()



    def __init_optimizer(self):
        pose_opt_params = []
        for viewpoint_cam in self.viewpoint_stack:
            pose_opt_params.append(
                {
                    "params": [viewpoint_cam.cam_rot_delta],
                    "lr": 0.003,
                    "name": "rot_{}".format(viewpoint_cam.uid),
                }
            )
            pose_opt_params.append(
                {
                    "params": [viewpoint_cam.cam_trans_delta],
                    "lr": 0.001,
                    "name": "trans_{}".format(viewpoint_cam.uid),
                }
            )
        self.pose_optimizer = torch.optim.Adam(pose_opt_params)
        self.pose_optimizer.zero_grad()



    def step(self):
        self.pose_optimizer.step()
        for viewpoint_cam in self.viewpoint_stack:
            if viewpoint_cam.uid != 0:
                update_pose(viewpoint_cam)



    def zero_grad(self):
        self.pose_optimizer.zero_grad()
        for viewpoint_cam in self.viewpoint_stack:
            viewpoint_cam.cam_rot_delta.data.fill_(0)
            viewpoint_cam.cam_trans_delta.data.fill_(0)


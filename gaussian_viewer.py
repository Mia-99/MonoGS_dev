
# https://github.com/isl-org/Open3D/blob/73508bcaba0a9a31e398bf8de76e3bbeaed81540/examples/python/visualization/video.py
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import time
import threading



import os

from gui.gl_render import util as util
from gui.gl_render import util_gau_lima as util_gau
from gui.gl_render import render_ogl as render_ogl


from gui.gui_utils import create_frustum, cv_gl

import cv2

from OpenGL import GL as gl
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
import json

import matplotlib.pyplot as plt

from gaussian_splatting.utils.graphics_utils import fov2focal, getWorld2View2



import torch



class CamInfo:
    def __init__(self, uid, R, T):
        self.uid = uid
        self.R = torch.from_numpy(R)
        self.T = torch.from_numpy(T)

    @property
    def camera_center(self):
        T = getWorld2View2(self.R, self.T).transpose(0, 1)
        return T.inverse()[3, :3]


def create_gaussians_gl (gaussians):
    gaussians_gl = util_gau.naive_gaussian()
    gaussians_gl.xyz = gaussians.get_xyz.detach().cpu().numpy()
    gaussians_gl.opacity = gaussians.get_opacity.detach().cpu().numpy()
    gaussians_gl.scale = gaussians.get_scaling.detach().cpu().numpy()
    gaussians_gl.rot = gaussians.get_rotation.detach().cpu().numpy()
    gaussians_gl.sh = gaussians.get_features.detach().cpu().numpy()[:, 0, :]
    return gaussians_gl






class Viewer:
    '''
        -- Mouse view control --
        Left button + drag         : Rotate.
        Ctrl + left button + drag  : Translate.
        Wheel button + drag        : Translate.
        Shift + left button + drag : Roll.
        Wheel                      : Zoom in/out.

        -- Keyboard view control --
        [/]          : Increase/decrease field of view.
        R            : Reset view point.
        Ctrl/Cmd + C : Copy current view status into the clipboard.
        Ctrl/Cmd + V : Paste view status from clipboard.

        -- General control --
        Q, Esc       : Exit window.
        H            : Print help message.
        P, PrtScn    : Take a screen capture.
        D            : Take a depth capture.
        O            : Take a capture of current rendering settings.   
    '''
    def __init__(self, viewpoint_stack = None, gaussians_gl = None, pose_vis_opts = 2):        

        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.viewpoint_stack = viewpoint_stack
        self.gaussians_gl = gaussians_gl

        self.g_scale_modifier = 1.0
        self.camera_size = 0.1

        '''
            Open3D Visualizer Example
            link: https://github.com/isl-org/Open3D/blob/73508bcaba0a9a31e398bf8de76e3bbeaed81540/examples/python/visualization/video.py                 
        '''

        self.WIDTH, self.HEIGHT = 600, 400

        self.window = gui.Application.instance.create_window ( "viewer: figure saved automatically", width=self.WIDTH, height=self.HEIGHT )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        # self.window.set_on_key(self._on_key)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d)

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "defaultLit"


        '''
            plot cameras a view furstums               
        '''
        if pose_vis_opts & 1:
            self.plot_trajectory(self.viewpoint_stack)

        '''
            plot the trajectory of camera centers               
        '''
        if pose_vis_opts & 2:
            self.plot_cameras(self.viewpoint_stack)


        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(45.0, bounds, bounds.get_center())
        self.widget3d.scene.show_axes(False)

        self.widget3d_width_ratio = 1.0
        self.widget3d_width = self.window.size.width

        """
        For visualize 3DGS ellipsoids
        """
        self.g_camera = util.Camera(h=self.HEIGHT, w=self.WIDTH)
        self.window_gl = self.init_glfw(self.WIDTH, self.HEIGHT)
        self.g_renderer = render_ogl.OpenGLRenderer(self.g_camera.w, self.g_camera.h)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)


        # get current camera view
        (W2C, FoVx, FoVy, fx, fy, cx, cy, H, W) = self.get_current_cam()

        # W2C = np.array(
        #                 [[-9.43868041e-01,  2.80348748e-01,  1.74693331e-01, -1.74692627e-02],
        #                 [-2.82218784e-01, -9.59239423e-01,  1.45645794e-02, -1.45645207e-03],
        #                 [ 1.71655729e-01, -3.55546921e-02,  9.84515131e-01,  5.70783520e+00],
        #                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
        # )
        # W = 600
        # H = 400
        # FoVy = 0.7853981633974483

        # set the camera view-control in case the view is user defined
        C2W = np.linalg.inv(W2C)
        frustum = create_frustum( C2W )
        viewpoint = frustum.view_dir
        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        self.render_img = self.render_o3d_image(W2C, FoVy, H, W)
        self.widget3d.scene.set_background([0, 0, 0, 1], self.render_img)

        self.save_figure()

        # a thread that helps to update view-control
        self.is_done = False
        threading.Thread(target=self._update_thread).start()

        app.run()
        glfw.terminate()


    def init_glfw(self, width, height):
        window_name = "headless rendering"
        if not glfw.init():
            exit(1)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(
            width, height, window_name, None, None
        )
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        if not window:
            glfw.terminate()
            exit(1)
        return window


    def plot_cameras(self, viewpoint_stack = None):
        if (viewpoint_stack is None):
            return
        for camera in viewpoint_stack:
            name = "cam{}_".format(camera.uid)
            self.add_camera(camera, name, color=[0, 1, 0], size=self.camera_size)


    def plot_trajectory(self, viewpoint_stack = None):
        if (viewpoint_stack is None):
            return
        odometry_line_set = self.create_trajectory_lineset(viewpoint_stack, color=[0, 0, 1])
        self.widget3d.scene.add_geometry("trajectory", odometry_line_set, self.lit)


    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.widget3d_width_ratio = 1.0
        self.widget3d_width = int(
            self.window.size.width * self.widget3d_width_ratio
        ) 
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )


    def _on_close(self):
        self.is_done = True       
        return True  # False would cancel the close
    

    # def _on_key(self, e):
    #     if e.key == gui.KeyName.S or e.key == gui.KeyName.s:
    #         self.save_figure()
    #     return True




    def _update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.
        while True:
            time.sleep(0.01)
            if self.is_done:
                o3d.visualization.gui.Application.instance.quit()

            # get current camera view
            (W2C, FoVx, FoVy, fx, fy, cx, cy, H, W) = self.get_current_cam()
            # ## compute_Gaussian_background here:
            self.render_img = self.render_o3d_image(W2C, FoVy, H, W)

            print(f"\ncurrent view info:")
            print(f"\tWIDTH = {W}, HEIGHT = {H}, FoVy = {FoVy}")
            print(f"\tW2C:\n{W2C}")

            # Update the images. This must be done on the UI thread.
            def update():
                self.widget3d.scene.set_background([0, 0, 0, 1], self.render_img)
                time.sleep(0.01)
                self.save_figure()

            gui.Application.instance.post_to_main_thread(self.window, update)
                        


    def save_figure(self):
        filename = "viewer_fig_automatic_save"
        height = self.window.size.height
        width = self.window.size.width
        app = o3d.visualization.gui.Application.instance
        img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}.png", img)



    def add_camera(self, camera, name, color=[0, 0, 1], size=0.01):
        W2C = getWorld2View2(camera.R, camera.T)
        W2C = W2C.cpu().numpy()
        C2W = np.linalg.inv(W2C)
        frustum = create_frustum(C2W, color, size=size)
        self.widget3d.scene.add_geometry(name, frustum.line_set, self.lit)   
        self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, True)
        return frustum


    def create_trajectory_lineset(self, viewpoint_stack, color=[0, 0, 1]):
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

    @staticmethod
    def vfov_to_hfov(vfov_deg, height, width):
        # http://paulbourke.net/miscellaneous/lens/
        return np.rad2deg(
            2 * np.arctan(width * np.tan(np.deg2rad(vfov_deg) / 2) / height)
        )

    def get_current_cam(self):
        w2c = cv_gl @ self.widget3d.scene.camera.get_view_matrix()
        image_gui = torch.zeros(
            (1, int(self.window.size.height), int(self.widget3d_width))
        )
        vfov_deg = self.widget3d.scene.camera.get_field_of_view()
        hfov_deg = self.vfov_to_hfov(vfov_deg, image_gui.shape[1], image_gui.shape[2])
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_gui.shape[2])
        fy = fov2focal(FoVy, image_gui.shape[1])
        cx = image_gui.shape[2] // 2
        cy = image_gui.shape[1] // 2
        H=image_gui.shape[1]
        W=image_gui.shape[2]
        return (w2c, FoVx, FoVy, fx, fy, cx, cy, H, W)



    def render_o3d_image(self, W2C, FoVy, H, W):

        WIDTH, HEIGHT = W, H
        self.window_gl  = self.init_glfw(WIDTH, HEIGHT)
        self.g_renderer = render_ogl.OpenGLRenderer(WIDTH, HEIGHT)
        # glfw.make_context_current(self.window_gl)
        
        glfw.poll_events()
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(
            gl.GL_COLOR_BUFFER_BIT
            | gl.GL_DEPTH_BUFFER_BIT
            | gl.GL_STENCIL_BUFFER_BIT
        )

        w = int(WIDTH * self.widget3d_width_ratio)
        glfw.set_window_size(self.window_gl, w, HEIGHT)

        C2W = np.linalg.inv(W2C)
        frustum = create_frustum( C2W )
        # viewpoint = frustum.view_dir
        # self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        self.g_camera.fovy = FoVy
        self.g_camera.update_resolution(HEIGHT, w)
        self.g_renderer.set_render_reso(w, HEIGHT)
        self.g_camera.position = frustum.eye.astype(np.float32)
        self.g_camera.target = frustum.center.astype(np.float32)
        self.g_camera.up = frustum.up.astype(np.float32)

        self.update_activated_renderer_state(self.gaussians_gl)
        self.g_renderer.sort_and_update(self.g_camera)
        width, height = glfw.get_framebuffer_size(self.window_gl)
        self.g_renderer.draw()
        bufferdata = gl.glReadPixels(
            0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
        )
        img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
        img = cv2.flip(img, 0)
        render_img = o3d.geometry.Image(img)
        glfw.swap_buffers(self.window_gl)

        return render_img


    def update_activated_renderer_state(self, gaus):
        self.g_renderer.update_gaussian_data(gaus)
        self.g_renderer.sort_and_update(self.g_camera)
        self.g_renderer.set_scale_modifier(self.g_scale_modifier)
        self.g_renderer.set_render_mod(-1-3) # original 7-3
        self.g_renderer.update_camera_pose(self.g_camera)
        self.g_renderer.update_camera_intrin(self.g_camera)
        self.g_renderer.set_render_reso(self.g_camera.w, self.g_camera.h)




def read_camera_json (json_file_path):
    """
    {"id": 0, "img_name": "IMG_6292", "width": 1332, "height": 876, "position": [-1.4759880629577484, 1.6090724813669521, -2.7727036587765035], "rotation": [[0.5408209248789425, -0.8404510054983934, -0.03398285699915072], [0.003746154845639685, 0.042807333797130434, -0.999076322658611], [0.8411294154510098, 0.540194075800467, 0.026299561462536303]], "fy": 1034.9718637370904, "fx": 1035.4965990500061}
    """
    cam_infos = []

    with open(json_file_path, 'r') as json_file:
        contents = json.load(json_file)


    for cam_info in contents:
        uid = cam_info["id"]
        img_name = cam_info["img_name"]
        W = cam_info["width"]
        H = cam_info["height"]
        T = cam_info["position"]
        R = cam_info["rotation"]
        fx = cam_info["fx"]
        fy = cam_info["fy"]

        # get the world-to-camera transform and set R, T
        R = np.array(R)
        T = np.array(T)

        gR = np.transpose(R)
        gT = - np.transpose(R) @ T

        cam = CamInfo(
            uid, gR, gT
        )
        cam_infos.append(cam)

    return cam_infos




def main():

    camera_file_path = "/hdd/3DGS/bicycle/cameras.json"
    point_cloud_file_path = "/hdd/3DGS/bicycle/point_cloud/iteration_7000/point_cloud.ply"

    cam_infos = read_camera_json (camera_file_path)
    gaussians_gl = util_gau.load_ply(point_cloud_file_path)


    Fig = Viewer(viewpoint_stack=cam_infos,  gaussians_gl= gaussians_gl)


if __name__ == "__main__":
    main()


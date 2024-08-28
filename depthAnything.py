
import torch
import numpy as np
import matplotlib
import cv2

import imgviz



from submodules.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2



class DepthAnything:

    def __init__(self, encoder = 'vitb') -> None:
        # encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }

        self.model = DepthAnythingV2(**self.model_configs[encoder])
        self.model.load_state_dict(torch.load(f'submodules/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = self.model.to('cuda').eval()

    def eval(self, raw_img):
        depth = self.model.infer_image(raw_img) # HxW raw depth map in numpy
        return depth
    
    @staticmethod
    # candidate colormaps: 'nipy_spectral', 'Spectral_r', 'turbo', 'plasma'
    def depth2image (depth, colormap=None):
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        if colormap is not None:
            cmap = matplotlib.colormaps.get_cmap(colormap)
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        return depth





if __name__ == "__main__":


    DA  = DepthAnything(encoder = 'vitl')

    image_dir = '/hdd/tandt/truck/images/000110.jpg'

    raw_img = cv2.imread(image_dir)

    depth = DA.eval(raw_img) # HxW raw depth map in numpy
    # depth = DA.depth2image(depth, colormap='nipy_spectral')

    depth = ( depth / np.median(depth) ) * 2.0 


    depth = imgviz.depth2rgb(depth, min_value=0.3, max_value=5.0)
    depth = torch.from_numpy(depth)
    depth = torch.permute(depth, (2, 0, 1)).float()
    depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    # rgb = o3d.geometry.Image(depth)


    cv2.imshow('raw-image', raw_img)
    cv2.imshow('depth-image', depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




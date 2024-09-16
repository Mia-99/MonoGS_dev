import torch
import torch.nn.functional as F

import pyscsp
import numpy as np

# https://github.com/tonylindeberg/pyscsp/blob/main/torchaffscsptest.ipynb
# https://github.com/tonylindeberg/pyscsp/blob/main/torchscsptest.ipynb

from gaussian_splatting.utils.loss_utils import l1_loss, ssim

from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt


import scipy







def image_conv_gaussian_separable (image : torch.Tensor, sigma=1, epsilon=0.01) -> torch.Tensor:
    """ Convolve an image (rgd, or gray) by Discrete Gassian kernel from Tony Lindeberg
    see. https://github.com/tonylindeberg/pyscsp    
    The implementation takes advantage of the fact that Gaussian kernel is separable.
    see. https://en.wikipedia.org/wiki/Separable_filter
    """
    channels = 1
    if image.ndim >= 3:
        channels = image.shape[-3]

    # Gaussian kernel form pyscsp
    sep1Dfilter = pyscsp.discscsp.make1Ddiscgaussfilter(
        sigma=sigma,
        epsilon=epsilon,
        D=image.ndim
    )
    kernel_1d = torch.from_numpy(sep1Dfilter).type(image.dtype).to(image.device)
    padding = len(kernel_1d) // 2

    # make 4D input
    if channels == 1:
        inputs = image.unsqueeze(0).unsqueeze(0)
    if channels == 3:
        inputs = image.unsqueeze(0)

    # add padding. use 'reflect' as in pyscsp
    p2d = (padding, padding, padding, padding)
    img = torch.nn.functional.pad(inputs, pad=p2d, mode='reflect')

    # Convolve along columns and rows. input 4D tensor
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), padding=0, bias=None, groups=channels)
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), padding=0, bias=None, groups=channels)

    if channels == 1:
        return img.squeeze_(0).squeeze_(0)
    if channels == 3:
        return img.squeeze_(0)






def image_conv_gaussian (image : torch.Tensor, sigma=1, epsilon=0.01) -> torch.Tensor:
    """ Convolve an image (rgd, or gray) by Discrete Gassian kernel from Tony Lindeberg
    see. https://github.com/tonylindeberg/pyscsp    
    This implementation is less efficient than exploiting the separable kernel.
    """
    channels = 1
    if image.ndim >= 3:
        channels = image.shape[-3]

    sep1Dfilter = pyscsp.discscsp.make1Ddiscgaussfilter(
        sigma=sigma,
        epsilon=epsilon,
        D=image.ndim
    )
    sep2Dfilter = np.outer(sep1Dfilter, sep1Dfilter)

    kernel_2d = torch.from_numpy(sep2Dfilter).type(image.dtype).to(image.device)
    padding = len(kernel_2d) // 2

    # (out_channels,  in_channels/groups,  kH,  kW)
    kernel_2d = kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)

    # (mini_batch,  in_channels,  iH,  iW)
    # padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    # Circular, replicate and reflection padding are implemented for the last 2 dimensions of a 3D or 4D input tensor
    # add padding. use 'reflect' as in pyscsp
    if channels == 1:
        inputs = image.unsqueeze(0).unsqueeze(0)
        p2d = (padding, padding, padding, padding)
        inputs = torch.nn.functional.pad(inputs, pad=p2d, mode='reflect', value=None)
        return F.conv2d(inputs, kernel_2d, padding=0, bias=None, groups=channels).squeeze_(0).squeeze_(0)

    if channels == 3:
        inputs = image.unsqueeze(0)
        p2d = (padding, padding, padding, padding)
        inputs = torch.nn.functional.pad(inputs, pad=p2d, mode='reflect', value=None)
        return F.conv2d(inputs, kernel_2d, padding=0, bias=None, groups=channels).squeeze_(0)






# for test
def image_conv_gaussian_1_channel (image : torch.Tensor, sigma=1, epsilon=0.1) -> torch.Tensor:
    # this method returns a np.array
    sep1Dfilter = pyscsp.discscsp.make1Ddiscgaussfilter(
        sigma=sigma,
        epsilon=epsilon,
        D=image.ndim
    )
    kernel_1d = torch.from_numpy(sep1Dfilter).type(image.dtype).to(image.device)
    padding = len(kernel_1d) // 2

    img = image.unsqueeze(0).unsqueeze_(0)

    p2d = (padding, padding, padding, padding)
    img = torch.nn.functional.pad(img, pad=p2d, mode='reflect')

    # Convolve along columns and rows. input 4D tensor
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding=0)
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding=0)
    return img.squeeze_(0).squeeze_(0)  # Make 2D again




# for test
def conv_gaussian_1d (input : torch.Tensor, sigma=1, epsilon=0.1) -> torch.Tensor:
    # this method returns a np.array
    sep1Dfilter = pyscsp.discscsp.make1Ddiscgaussfilter(
        sigma=sigma,
        epsilon=epsilon,
        D=1
    )
    kernel_1d = torch.from_numpy(sep1Dfilter).type(input.dtype).to(input.device)
    padding = len(kernel_1d) // 2

    img = input.unsqueeze(0).unsqueeze_(0)

    p1d = (padding, padding)
    img= torch.nn.functional.pad(img, pad=p1d, mode='constant', value=0.0)
    print(f"torch.nn.functional.pad: \n{img}")

    img = F.conv1d(img, weight=kernel_1d.view(1, 1, -1), padding=0)
    print(f"torch.nn.functional.cov1d: \n{img}")

    return img.squeeze_(0).squeeze_(0)  # Make 2D again





if __name__ == "__main__":



    image_name = "/hdd/tandt/truck/images/000007.jpg"


    img = Image.open(image_name)
    img_np = np.array(img)

    print(f"img.size = {img.size}")
    print(f"img_np.shape = {img_np.shape}")

    coldala = plt.imread(image_name)
    image = torch.from_numpy(coldala.transpose(2, 0, 1)).type(torch.float64)

    print(f"numpy image shape: {coldala.shape}")
    print(f"tensor image shape: {image.shape}")



    for D in np.arange(1, 7):
        sep1Dfilter = pyscsp.discscsp.make1Ddiscgaussfilter(
            sigma=4,
            epsilon=0.01,
            D=D
        )
        print(f"sep1Dfilter (D = {D}) = \n {sep1Dfilter}")





    smoothpic = pyscsp.discscsp.discgaussconv(
        inpic = coldala,
        sigma = 4,
        epsilon = 0.01
    )
    smoothpic_c1 = pyscsp.discscsp.discgaussconv(
        inpic = coldala[:,:,0],
        sigma = 4,
        epsilon = 0.01
    )
    smoothpic_c2 = pyscsp.discscsp.discgaussconv(
        inpic = coldala[:,:,1],
        sigma = 4,
        epsilon = 0.01
    )
    smoothpic_c3 = pyscsp.discscsp.discgaussconv(
        inpic = coldala[:,:,2],
        sigma = 4,
        epsilon = 0.01
    )
    print(f"test pyscsp.discscsp.discgaussconv. error in channel 1: {np.linalg.norm(smoothpic[:,:,0] - smoothpic_c1)}")
    print(f"test pyscsp.discscsp.discgaussconv. error in channel 2: {np.linalg.norm(smoothpic[:,:,1] - smoothpic_c2)}")
    print(f"test pyscsp.discscsp.discgaussconv. error in channel 3: {np.linalg.norm(smoothpic[:,:,2] - smoothpic_c3)}")





    conv_img = image_conv_gaussian(image, sigma=4, epsilon = 0.01)
    conv_img_sep = image_conv_gaussian_separable(image, sigma=4, epsilon = 0.01)
    print(f"test 2D separable convolution and 2D convolution difference.\n\terror in total: {torch.norm(conv_img - conv_img_sep)}")
    print(f"\terror in channel 1: {torch.norm(conv_img[0] - conv_img_sep[0])}")
    print(f"\terror in channel 2: {torch.norm(conv_img[1] - conv_img_sep[1])}")
    print(f"\terror in channel 3: {torch.norm(conv_img[2] - conv_img_sep[2])}")



    conv_img1 = image_conv_gaussian(image[0], sigma=4, epsilon = 0.01)
    conv_img2 = image_conv_gaussian(image[1], sigma=4, epsilon = 0.01)
    conv_img3 = image_conv_gaussian(image[2], sigma=4, epsilon = 0.01)

    # print(f"test 2D convolution. error in channel 1: {torch.norm(conv_img[0] - conv_img1)}")
    # print(f"test 2D convolution. error in channel 2: {torch.norm(conv_img[1] - conv_img2)}")
    # print(f"test 2D convolution. error in channel 3: {torch.norm(conv_img[2] - conv_img3)}")


    conv_img_c1 = image_conv_gaussian_separable(image[0], sigma=4, epsilon = 0.01)
    conv_img_c2 = image_conv_gaussian_separable(image[1], sigma=4, epsilon = 0.01)
    conv_img_c3 = image_conv_gaussian_separable(image[2], sigma=4, epsilon = 0.01)


    print(f"test 1D (of separable kernel) and 2D convolution difference. error in channel 1: {torch.norm(conv_img_c1 - conv_img1)}")
    print(f"test 1D (of separable kernel) and 2D convolution difference. error in channel 2: {torch.norm(conv_img_c2 - conv_img2)}")
    print(f"test 1D (of separable kernel) and 2D convolution difference. error in channel 3: {torch.norm(conv_img_c3 - conv_img3)}")


    smoothpic_c1_tensor = torch.from_numpy(smoothpic_c1).type(torch.FloatTensor)
    smoothpic_c2_tensor = torch.from_numpy(smoothpic_c2).type(torch.FloatTensor)
    smoothpic_c3_tensor = torch.from_numpy(smoothpic_c3).type(torch.FloatTensor)

    print(f"torch conv 1D and pyscsp.discscsp.discgaussconv. error in channel 1: {torch.norm(smoothpic_c1_tensor - conv_img_c1)}")
    print(f"torch conv 1D and pyscsp.discscsp.discgaussconv. error in channel 2: {torch.norm(smoothpic_c2_tensor - conv_img_c2)}")
    print(f"torch conv 1D and pyscsp.discscsp.discgaussconv. error in channel 3: {torch.norm(smoothpic_c3_tensor - conv_img_c3)}")


    print(f"torch conv 2D and pyscsp.discscsp.discgaussconv. error in channel 1: {torch.norm(smoothpic_c1_tensor - conv_img1)}")
    print(f"torch conv 2D and pyscsp.discscsp.discgaussconv. error in channel 2: {torch.norm(smoothpic_c2_tensor - conv_img2)}")
    print(f"torch conv 2D and pyscsp.discscsp.discgaussconv. error in channel 3: {torch.norm(smoothpic_c3_tensor - conv_img3)}")


    diff1 = torch.abs(smoothpic_c1_tensor - conv_img1).cpu().numpy()
    diff2 = torch.abs(smoothpic_c2_tensor - conv_img2).cpu().numpy()
    diff3 = torch.abs(smoothpic_c3_tensor - conv_img3).cpu().numpy()

    plt.imshow(diff1, cmap="gray")
    plt.colorbar()
    plt.show()
    plt.imshow(diff2, cmap="gray")
    plt.colorbar()
    plt.show()
    plt.imshow(diff3, cmap="gray")
    plt.colorbar()
    plt.show()




    # simple 2D data


    data_buffer = np.linspace(1, 63, num=63, dtype=np.float64)
    data_2d = data_buffer.reshape((9, 7))
    data_2d_tensor = torch.from_numpy(data_2d).type(torch.float64)


    pytorch_conv = image_conv_gaussian_1_channel(data_2d_tensor, sigma=1, epsilon = 0.1)
    pytorch_conv_2 = image_conv_gaussian(data_2d_tensor, sigma=1, epsilon = 0.1)
    pytorch_conv_11 = image_conv_gaussian_separable(data_2d_tensor, sigma=1, epsilon = 0.1)


    pyscsp_conv = pyscsp.discscsp.discgaussconv(
        inpic = data_2d,
        sigma = 1,
        epsilon = 0.1
    )
    pyscsp_conv_tensor = torch.from_numpy(pyscsp_conv).type(torch.float64)


    print(f"data_2d_tensor = \n {data_2d_tensor}")
    print(f"pytorch conv = \n {pytorch_conv}")
    print(f"pytorch conv_2 = \n {pytorch_conv_2}")
    print(f"pytorch conv_11 = \n {pytorch_conv_11}")
    print(f"pyscsp conv = \n {pyscsp_conv_tensor}")
    print(f"difference = \n\t 1_channel_sep - pyscsp_conv \n\t\t {np.abs(pyscsp_conv_tensor - pytorch_conv) > 1e-5}")    
    print(f"\t 1_channel_sep - conv_2d \n\t\t{pytorch_conv - pytorch_conv_2}")   
    print(f"\t 1_channel_sep - conv_2d_separable \n\t\t{pytorch_conv - pytorch_conv_11}")   





    # simple 1D data



    data_buffer = np.linspace(1, 10, 10)
    data_buffer_torch = torch.from_numpy(data_buffer).type(torch.float64)

    conv1d = conv_gaussian_1d (data_buffer_torch, sigma=1, epsilon=0.1)
    print(f"pytorch_conv1d = \n{conv1d.cpu().numpy()}")

    sep1Dfilter = pyscsp.discscsp.make1Ddiscgaussfilter(
        sigma=1,
        epsilon=0.1,
        D=1
    )
    scipy_conv1d = scipy.ndimage.correlate1d(data_buffer, weights=sep1Dfilter, axis=0, output=None, mode='constant', cval=0.0, origin = 0)
    print(f"scipy_conv1d = \n{scipy_conv1d}")


    print(f"with constant = 0 paddings.  \n\tscipy_conv1d - conv1d:  {np.linalg.norm(conv1d.cpu().numpy() - scipy_conv1d)}")


    pyscsp_conv1d = pyscsp.discscsp.discgaussconv(
        inpic = data_buffer,
        sigma = 1,
        epsilon = 0.1
    )
    print(f"pyscsp_conv1d = \n{pyscsp_conv1d}")


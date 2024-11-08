import numpy as np
from matplotlib import pyplot as plt
from math import pi
from matplotlib.patches import Ellipse

from numpy import linalg as LA


from matplotlib import style
print(plt.style.available)
# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 
# 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 
# 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 
# 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 
# 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 
# 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']






def pltCovMat2D (ax, mu, Sigma, edgecolor = 'k',faceColor = 'None' ):
    a = Sigma[0, 0]
    b = Sigma[0, 1]
    c = Sigma[1, 1]
    cont = np.sqrt( ((a-c)/2.0)**2 + b**2.0 )

    lambda1 = (a+c)/2.0 + cont
    lamdba2 = (a+c)/2.0 - cont
    theta = np.arctan2(lambda1 - a, b) * 180 / np.pi

    ellipse = Ellipse(xy=mu, width=lambda1, height=lamdba2, angle=theta,
                        edgecolor=edgecolor, fc=faceColor, lw=2)
    ax.add_patch(ellipse)
    return ax
    # eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    # eigvec1 = eigenvectors[:, 1]
    # eigvec2 = eigenvectors[:, 2]
    # eigvec3 = eigenvectors[:, 3]


def distortByRadial1stOrder (mu, Sigma = None, kappa = 0.01):
    x = mu[0]
    y = mu[1]
    c = 1.0 + kappa*(x*x + y*y)
    if Sigma is not None:
        J = c * np.eye(2) + 2.0 * kappa * np.array([ [x*x,  x*y],  [x*y,  y*y] ])
        return c*mu,  J @ Sigma @ np.transpose(J)  
    else:
        return c*mu



if __name__ == "__main__":


    # command on styles must go first to take effect
    plt.style.use('seaborn-paper')
    # plt.style.use('dark_background')
    # plt.style.use('fivethirtyeight')
    # plt.style.use('ggplot')
    # plt.style.use('bmh')

    size = 10
    params = {'legend.fontsize': 'large',
            'figure.figsize': (5.5, 5.5),
            'axes.labelsize': size,
            'axes.titlesize': size*1.5,
            'xtick.labelsize': size*0.75,
            'ytick.labelsize': size*0.75,
            'axes.titlepad': 10}
    plt.rcParams.update(params)



    kappa = -0.05

    color_original_points = 'gray'
    color_distorted_points = 'g'
    color_distortion_field = 'r'

    color_ellipse_original = 'k'
    color_ellipse_distorted = 'b'


    nx, ny = (9, 9)
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    xv, yv = np.meshgrid(x, y)

    # 1st order radial distortion
    u = xv * ( kappa * (xv**2 + yv**2) )
    v = yv * ( kappa * (xv**2 + yv**2) )
    xv_new = xv + u
    yv_new = yv + v
    



    fig = plt.figure()
    ax = fig.gca()
    ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal")





    ax.plot(xv, yv, marker='.', color=color_original_points, linestyle='none')
    # ax.plot(xv_new, yv_new, marker='.', color=color_distorted_points, linestyle='none')
    plt.quiver(x,y,u,v, color=color_distortion_field, units='xy', scale=1.0)


    # ellipse 1
    mu = np.array([0.5, 0.5])
    Sigma = 0.5*np.array([ [1, 0.5], [0.5, 1.5] ])
    
    print(f"Original:  mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax, mu, Sigma, color_ellipse_original)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax, mu, Sigma, color_ellipse_distorted)


    # ellipse 2
    mu = np.array([-0.5, -0.65])
    Sigma = 0.5*np.array([ [1.5, -0.5], [-0.5, 0.5] ])
    
    print(f"Original:  mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax, mu, Sigma, color_ellipse_original)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax, mu, Sigma, color_ellipse_distorted)


    # ellipse 3
    mu = np.array([-0.5, 0.5])
    Sigma = 0.5*np.array([ [1.5, 0], [0, 0.5] ])
    
    print(f"Original:  mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax, mu, Sigma, color_ellipse_original)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax, mu, Sigma, color_ellipse_distorted)



    # ellipse 4
    mu = np.array([0.7, -0.5])
    Sigma = 0.5*np.array([ [0.5, 0], [0, 1.5] ])
    
    print(f"Original:  mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax, mu, Sigma, color_ellipse_original)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax, mu, Sigma, color_ellipse_distorted)




    # ellipse 5
    mu = np.array([-0., 0.])
    Sigma = 0.5*np.array([ [1.5, -0.5], [-0.5, 0.5] ])
    
    print(f"Original:  mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax, mu, Sigma, color_ellipse_original)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax, mu, Sigma, color_ellipse_distorted)



    ax.set_title(f'$\kappa = {kappa}$')


    plt.tight_layout()
    plt.savefig("gaussian_distortion.pdf")    
    plt.show(block=True)
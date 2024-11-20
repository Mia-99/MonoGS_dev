import numpy as np
from matplotlib import pyplot as plt
from math import pi
from matplotlib.patches import Ellipse

from numpy import linalg as LA
from matplotlib import style
print(plt.style.available)




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

    line_length = 0.2
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    # print(f"eigenvectors: {eigenvectors}")
    eigvec1 = eigenvectors[:, 0]
    eigvec2 = eigenvectors[:, 1]
    vec1_length = line_length * np.sqrt(eigenvalues[0])
    vec2_length = line_length * np.sqrt(eigenvalues[1])
    line1_x = [mu[0] - eigvec1[0] * vec1_length, mu[0] + eigvec1[0] * vec1_length]
    line1_y = [mu[1] - eigvec1[1] * vec1_length, mu[1] + eigvec1[1] * vec1_length]
    

    # Minor axis line
    line2_x = [mu[0] - eigvec2[0] * vec2_length, mu[0] + eigvec2[0] * vec2_length]
    line2_y = [mu[1] - eigvec2[1] * vec2_length, mu[1] + eigvec2[1] * vec2_length]
    
    return ax, np.array([line1_x, line1_y]), np.array([line2_x, line2_y])

def distortByRadial1stOrder (mu, Sigma = None, kappa = 0.01):
    x = mu[0]
    y = mu[1]
    # c = 1.0 + kappa*(xv + yv)
    c = 1.0 + kappa*(x + y)
    if Sigma is not None:
        J = c * np.eye(2) + 2.0 * kappa * np.array([ [x,  x],  [x,  y] ])
        # J = c * np.eye(2) + 2.0 * kappa * np.array([ [xv,  xv],  [xv,  yv] ])
        return c*mu,  J @ Sigma @ np.transpose(J)  
    else:
        return c*mu



if __name__ == "__main__":
    plt.style.use('ggplot')

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
    kappa_neg = -kappa
    # kappa = -0.5

    # 
    dark_red = '#8B0000'
    light_red = '#FF6347'
    light_light_red = '#FFA07A'
    dark_blue = '#00008B'
    light_blue = '#ADD8E6'
    light_light_blue = '#B0E0E6'

    color_original_points = light_red
    color_distorted_points = dark_red
    color_distortion_field = light_light_red

    color_ellipse_original = light_blue 
    color_ellipse_distorted = dark_blue


    nx, ny = (9, 9)
    x = np.linspace(-1.0, 1.0, nx)
    print(f"x = {x}")
    y = np.linspace(-1.0, 1.0, ny)
    xv, yv = np.meshgrid(x, y)

    # 1st order radial distortion
    u = xv * ( kappa * (xv**2 + yv**2) )
    v = yv * ( kappa * (xv**2 + yv**2) )
    xv_new = xv + u
    yv_new = yv + v

    u = xv * ( kappa_neg * (xv**2 + yv**2) )
    v = yv * ( kappa_neg * (xv**2 + yv**2) )
    xv_new_neg = xv + u
    yv_new_neg = yv + v
    

    alpha = 0.5
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal')
    # ax1.plot(xv, yv, color=color_original_points, linestyle='-', alpha=alpha)
    # ax1.plot(np.transpose(xv), np.transpose(yv), color=color_original_points, linestyle='-', alpha=alpha)
    ax1.plot(xv_new, yv_new, color=color_distorted_points, linestyle='-', alpha=alpha) # horizontal lines
    ax1.plot(np.transpose(xv_new), np.transpose(yv_new), color=color_distorted_points, linestyle='-', alpha=alpha) # vertical lines

    ax2 = fig.add_subplot(122)
    ax2.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal')
    # ax2.plot(xv, yv, color=color_original_points, linestyle='-', alpha=alpha)
    # ax2.plot(np.transpose(xv), np.transpose(yv), color=color_original_points, linestyle='-', alpha=alpha)
    ax2.plot(xv_new_neg, yv_new_neg, color=color_distorted_points, linestyle='-', alpha=alpha) # horizontal lines
    ax2.plot(np.transpose(xv_new_neg), np.transpose(yv_new_neg), color=color_distorted_points, linestyle='-', alpha=alpha) # vertical lines
    

    # ellipse 1
    mu = np.array([0.5, 0.5])
    Sigma = 0.5*np.array([ [1, 0.5], [0.5, 1.5] ])
    
    print(f"Original:  mu = {mu}\nSigma = \n{Sigma}")
    _, line1, line2= pltCovMat2D(ax1, mu, Sigma, color_ellipse_original)
    _, line1, line2= pltCovMat2D(ax2, mu, Sigma, color_ellipse_original)
    ax1.plot(line1[0], line1[1], color=color_ellipse_original, linestyle='-', alpha=0.5)
    ax1.plot(line2[0], line2[1], color=color_ellipse_original, linestyle='-', alpha=0.5)
    ax2.plot(line1[0], line1[1], color=color_ellipse_original, linestyle='-', alpha=0.5)
    ax2.plot(line2[0], line2[1], color=color_ellipse_original, linestyle='-', alpha=0.5)

    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    _, line1, line2 = pltCovMat2D(ax1, mu, Sigma, color_ellipse_distorted)
    ax1.plot(line1[0], line1[1], color=color_ellipse_distorted, linestyle='-', alpha=0.5)
    ax1.plot(line2[0], line2[1], color=color_ellipse_distorted, linestyle='-', alpha=0.5)

    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa_neg)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    _, line1, line2 = pltCovMat2D(ax2, mu, Sigma, color_ellipse_distorted)
    ax2.plot(line1[0], line1[1], color=color_ellipse_distorted, linestyle='-', alpha=0.5)
    ax2.plot(line2[0], line2[1], color=color_ellipse_distorted, linestyle='-', alpha=0.5)

    # ax.quiver([mu[0], mu[0]], [mu[1], mu[1]], 
    #         [distort_eigen_vector1[0], distort_eigen_vector2[0]], [distort_eigen_vector1[1], distort_eigen_vector2[1]], 
    #         color=color_ellipse_distorted, units='xy', scale=1, angles='xy')


    # ellipse 2
    mu = np.array([-0.5, -0.65])
    Sigma = 0.5*np.array([ [1.5, -0.5], [-0.5, 0.5] ])
    
    print(f"Original:  mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax1, mu, Sigma, color_ellipse_original)
    pltCovMat2D(ax2, mu, Sigma, color_ellipse_original)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax1, mu, Sigma, color_ellipse_distorted)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa_neg)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax2, mu, Sigma, color_ellipse_distorted)


    # ellipse 3
    mu = np.array([-0.5, 0.5])
    Sigma = 0.5*np.array([ [1.5, 0], [0, 0.5] ])
    
    print(f"Original:  mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax1, mu, Sigma, color_ellipse_original)
    pltCovMat2D(ax2, mu, Sigma, color_ellipse_original)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax1, mu, Sigma, color_ellipse_distorted)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa_neg)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax2, mu, Sigma, color_ellipse_distorted)


    # ellipse 4
    mu = np.array([0.7, -0.5])
    Sigma = 0.5*np.array([ [0.5, 0], [0, 1.5] ])
    
    print(f"Original:  mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax1, mu, Sigma, color_ellipse_original)
    pltCovMat2D(ax2, mu, Sigma, color_ellipse_original)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax1, mu, Sigma, color_ellipse_distorted)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa_neg)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax2, mu, Sigma, color_ellipse_distorted)



    # ellipse 5
    mu = np.array([-0., 0.])
    Sigma = 0.5*np.array([ [1.5, -0.5], [-0.5, 0.5] ])
    
    print(f"Original:  mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax1, mu, Sigma, color_ellipse_original)
    pltCovMat2D(ax2, mu, Sigma, color_ellipse_original)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax1, mu, Sigma, color_ellipse_distorted)
    mu, Sigma = distortByRadial1stOrder (mu, Sigma, kappa_neg)
    print(f"Distorted: mu = {mu}\nSigma = \n{Sigma}")
    pltCovMat2D(ax2, mu, Sigma, color_ellipse_distorted)

    # plt.title(f'$\kappa$ = {kappa}')
    ax1.set_title(f'$\kappa$ = {kappa}')
    ax2.set_title(f'$\kappa$ = {kappa_neg}')

    plt.show()
import torch
import matplotlib.pyplot as plt
import numpy as np


def gaussian1D(M, mean, sigma, a=1):
    '''1D gaussian vector'''
    # w = 1/(sigma * np.sqrt(2*np.pi)) * torch.exp(-(1/2) * ((x - mean)/sigma)**2)
    a = a
    # x = torch.arange(0, M) - (M - 1.0) / 2.0 ## don't substract by 1
    x = torch.arange(0, M) - (M / 2)
    var = 2 * sigma**2
    w = a * torch.exp(-((x - mean)**2 / var))
    return w


def gaussian2D(kernlen=50, mean=0, sigma=5, a=1):
    '''outer product of two gaussian vectors'''
    gkernel1d = gaussian1D(kernlen, mean=mean, sigma=sigma, a=a)
    gkernel2d = torch.outer(gkernel1d, gkernel1d)
    return gkernel2d


def swap_quadrants(input_volume):
    """FFT returns features centered around the origin not the center of the image"""
    L = input_volume.size(-1)
    L2 = int(L / 2)
    output_volume = torch.zeros(input_volume.shape)

    output_volume[:L2, :L2] = input_volume[L2:L, L2:L]
    output_volume[L2:L, L2:L] = input_volume[:L2, :L2]

    output_volume[L2:L, :L2] = input_volume[:L2, L2:L]
    output_volume[:L2, L2:L] = input_volume[L2:L, :L2]

    output_volume[L2:L, L2:L] = input_volume[:L2, :L2]
    output_volume[:L2, :L2] = input_volume[L2:L, L2:L]

    return output_volume


if __name__ == '__main__':
    ### initialize two different 2D gaussians
    sigma1, sigma2 = 3, 4
    mean1, mean2 = -5, 5
    amp1, amp2 = 1, 1
    ampcheck = amp1*amp2
    boxsize = 50
    cmap = 'gray'
    vmin, vmax = 0,1
    gaussian_input1 = gaussian2D(boxsize, mean=mean1, sigma=sigma1, a=amp1)
    gaussian_input2 = gaussian2D(boxsize, mean=mean2, sigma=sigma2, a=amp2)

    ### Torch v1.10 FFT calls
    ## Normalization
    # norm = None
    norm = 'ortho'
    # norm = 'forward'
    # norm = 'backward'

    cplx_G1 = torch.fft.rfft2(gaussian_input1, dim=(-2, -1), norm=norm)
    cplx_G2 = torch.fft.rfft2(gaussian_input2, dim=(-2, -1), norm=norm)
    # gaussian_FFT = torch.fft.irfft2(torch.conj(cplx_G1) * torch.conj(cplx_G2), dim=(-2, -1), norm=norm) ## this performs a convolution of the two shapes
    # gaussian_FFT = torch.fft.irfft2(cplx_G1 * cplx_G2, dim=(-2, -1), norm=norm) ## this also performs a convolution of the two shapes
    gaussian_FFT = torch.fft.irfft2(cplx_G1 * torch.conj(cplx_G2), dim=(-2, -1), norm=norm) ## this performs a proper correlation (what we want)

    ### Plotting 2D Gaussian inputs
    fig, ax = plt.subplots(1,5, figsize=(20,5))
    g1 = ax[0].imshow(gaussian_input1, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title('Guassian1 '+'$\mu_1$='+str(mean1)+' $\sigma_1=$'+str(sigma1))
    g2 = ax[1].imshow(gaussian_input2, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title('Guassian2 '+'$\mu_2$='+str(mean2)+' $\sigma_1=$'+str(sigma2))

    ### Plotting Convolution Output
    convolution = swap_quadrants(gaussian_FFT)
    conv = ax[2].imshow(convolution, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2].set_title(r'Gaussian1 $\bigstar$ Gaussian2')

    ### Checking with output distribution of ~N(mean1+mean2, sigma1^2 + sigma2^2)
    result_sigma = np.sqrt(sigma1**2+sigma2**2)
    # result_mean = mean1 + mean2 ## used sum for convolution
    result_mean = mean1 - mean2 ## use difference for correlation (correlation == -convolution)

    gaussian_check = gaussian2D(boxsize, mean=result_mean, sigma=result_sigma, a=ampcheck)
    scaled_gaussiancheck = gaussian_check
    gaussian_summedvariance = ax[3].imshow(scaled_gaussiancheck, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[3].set_title('Check '+' $\mu_1+\mu_2$='+str(mean1+mean2) + ' $\sqrt{\sigma_1^2 + \sigma_2^2}=$'+str(result_sigma)[:3])

    ## Difference between convolution output and gaussian check
    diff = ax[4].imshow(convolution - gaussian_check, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[4].set_title(r'Convolution - expected Gaussian')

    ax[2].grid(color='w')
    ax[3].grid(color='w')

    plt.show()

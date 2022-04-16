import torch
import matplotlib.pyplot as plt
import numpy as np


def gaussian_fn(M, sigma):
    x = torch.arange(0, M) - (M - 1.0) / 2.0
    var = 2 * sigma**2
    w = torch.exp(-x ** 2 / var)
    return w

def gkern(kernlen=50, sigma=5):
    gkern1d = gaussian_fn(kernlen, sigma=sigma)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d


def swap_quadrants(input_volume):
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
    sigma1 = 3
    sigma2 = 4
    boxsize = 50
    gaussian_input1 = gkern(boxsize, sigma=sigma1)
    gaussian_input2 = gkern(boxsize, sigma=sigma2)
    # plt.imshow(gaussian_filter, cmap='gray')
    # plt.colorbar()
    # plt.show()

    cplx_G1 = torch.fft.rfft2(gaussian_input1, dim=(-2, -1))
    cplx_G2 = torch.fft.rfft2(gaussian_input2, dim=(-2, -1))
    gaussian_FFT = torch.fft.irfft2(cplx_G1 * torch.conj(cplx_G2), dim=(-2, -1))

    fig, ax = plt.subplots(1,4, figsize=(12,5))
    g1 = ax[0].imshow(gaussian_input1, cmap='gray')
    ax[0].set_title(r'Guassian1 $\sigma_1=$'+str(sigma1))
    g2 = ax[1].imshow(gaussian_input2, cmap='gray')
    ax[1].set_title(r'Guassian2 $\sigma_2=$'+str(sigma2))

    convolution = swap_quadrants(gaussian_FFT)
    conv = ax[2].imshow(convolution, cmap='gray')
    ax[2].set_title(r'Gaussian1 $\bigstar$ Gaussian2')

    result_sigma = np.sqrt(sigma1**2+sigma2**2)
    gaussian_check = gkern(boxsize, sigma=result_sigma)
    check = ax[3].imshow(gaussian_check, cmap='gray')
    ax[3].set_title('Check' + r'$\sqrt{\sigma_1^2 + \sigma_2^2}=$'+str(result_sigma)[:3])

    # plt.colorbar()
    plt.show()

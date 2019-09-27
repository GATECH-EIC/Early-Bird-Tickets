import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave, imread
from scipy.ndimage import fourier_gaussian
from PIL import Image
"""
Gaussian filter via frequency domain methods
We use '1 - template' to get the highpass filter template, the core idea is ifft(fft(img) .* template)
Note that for high frequency components, we focus on the edge information, in which way we use gray image for highpass filter and then map it back to three dimensional one for the convinence of inference.
"""

def Gaussian(src, sigma, ftype):
    h, w = src.shape
    template = np.zeros(src.shape, dtype=np.float32)
    d0 = 1 / (2 * np.pi * sigma) * h
    for i in np.arange(h):
        for j in np.arange(w):
            distance2 = (i - h / 2) ** 2 + (j - w / 2) ** 2
            template[i, j] = np.e ** (-1 * (distance2 / (2 * d0 ** 2)))
    if ftype == 'highpass':
        template = 1 - template
    return template

def rescale(x, lo, hi):
    """Rescale a tensor to [lo,hi]."""
    assert(lo < hi), "[rescale] lo={0} must be smaller than hi={1}".format(lo,hi)
    old_width = np.max(x)-np.min(x)
    old_center = np.min(x) + (old_width / 2.)
    new_width = float(hi-lo)
    new_center = lo + (new_width / 2.)
    # shift everything back to zero:
    x = x - old_center
    # rescale to correct width:
    x = x * (new_width / old_width)
    # shift everything to the new center:
    x = x + new_center
    # return:
    return x

def filter(img, sigma, mode='highpass'):
    # only support single-channel images
    template = Gaussian(img, sigma, mode)
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    img_hp_fft = np.multiply(img_fft, template)
    img_hp_fft = np.fft.ifftshift(img_hp_fft)
    img_hp = np.real(np.fft.ifft2(img_hp_fft))
    return rescale(img_hp, 0, 1)

def rgb2gray(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# slow version
def my_gaussian_filter(img, sigma, mode='highpass'):
    img = np.asarray(img)
    # img = img.convert('L')
    gray = rgb2gray(img)
    img_hp = filter(gray, sigma, mode)
    img_hp = np.stack((img_hp,)*3, axis=-1)
    # return img_hp
    return Image.fromarray(np.uint8(img_hp * 255))

# scipy version (cython accelerator)
def my_gaussian_filter_2(img, sigma, mode='highpass'):
    img = np.asarray(img.convert('L'))
    img_fft = np.fft.fft2(img)
    G = fourier_gaussian(img_fft, sigma)
    if mode == 'highpass':
        img_g = rescale(np.real(np.fft.ifft2(img_fft - G)), 0, 1)
    elif mode == 'lowpass':
        img_g = rescale(np.real(np.fft.ifft2(G)), 0, 1)
    else:
        print('no such mode!')
        return None
    img_g = np.stack((img_g,)*3, axis=-1)
    return Image.fromarray(np.uint8(img_g * 255))


if __name__ == '__main__':
    # img = np.random.randn(32,32, 3)
    # img = imread('dots.png', mode='RGB')
    img = Image.open('dots.png')
    img_hp = my_gaussian_filter_2(img, sigma=1, mode='highpass')
    print(img_hp.size)
    plt.figure()
    plt.imshow(img_hp)
    plt.savefig('dot_hp.png')

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors
import os


def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        pass
    else:
        raise ValueError('wavelength must be between 380 and 750')
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return R, G, B


def draw_line(img, ref, wavelength, areas, savedir):
    ch = img.shape[-1]
    corr_list = []

    for i, area in enumerate(areas):
        fig = plt.figure(figsize=(10, 10))

        value_pred = []
        value_gt = []

        for j in range(ch):
            value_pred.append(np.mean(img[area[1]:area[3], area[0]:area[2], j]))
            value_gt.append(np.mean(ref[area[1]:area[3], area[0]:area[2], j]))

        corr = np.corrcoef(np.array(value_pred), np.array(value_gt))
        corr_list.append(corr[0,1])
    
        plt.plot(wavelength, np.array(value_pred) / max(value_pred), label="pred", linewidth=4, marker="o", markersize=10)
        plt.plot(wavelength, np.array(value_gt) / max(value_gt), label="gt", linewidth=4, marker=".", markersize=10)
        plt.text(x=wavelength[0], y=0.8, s=f"corr={corr[0,1]:.4f}", fontsize=24)

        plt.xlabel('Wavelength (nm)', fontsize=24)
        plt.ylabel('Itensity', fontsize=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.legend(fontsize=24)
        plt.savefig(f'{savedir}/spectral_line_{i+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    return corr_list

def draw_cubes(img, wavelengths, savepath):
    """
    1. Draw and save image of each channel of Hyperspectral image with color map according to wavelength.
    2. Draw and save all channels of Hyperspectral image in one image, with wavelength text in left-top of each sub-image

    img: Hyperspectral image which shape should be (H, W, C). e.g (512, 512, 31).
    wavelengths: List of wavelength of each channel of Hyperspectral images.
    savedir: The path to store images. 
    """
    if img.shape[-1] != len(wavelengths):
        raise ValueError("Channels number must equal to length of array 'wavelengths'")

    line_num = int(np.ceil(len(wavelengths) / 7))
    fig = plt.figure(figsize=(30, line_num * 4))

    for i in range(len(wavelengths)):
        color = wavelength_to_rgb(wavelengths[i])
        colors = [(0, 0, 0), color]
        cmap = LinearSegmentedColormap.from_list('my_colormap', colors)

        plt.subplot(line_num, 7, i+1)
        img_temp = img[:, :, i]
        # vmin = np.min(img_temp)
        # vmax = np.max(img_temp)
        # img_temp = (img_temp - vmin) / (vmax - vmin)
        
        plt.imshow(img_temp, cmap=cmap, norm="linear")
        plt.text(20, 40, str(round(wavelengths[i], 1)) + " nm", fontsize=28, color="white", fontweight="bold")
        # plt.text(30, 80, str(round(wavelengths[i], 1)) + " nm", fontsize=14, color="white", fontweight="bold")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
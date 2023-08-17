import matplotlib.pyplot as plt
from spectral.io import envi

def load_h2o_data(mask_path: str):
    mask_header = envi.open(mask_path)
    mask_data = mask_header.open_memmap(interleave = 'bip')
    water_vapor = mask_data[:,:,6]
    return water_vapor

def plot_h2o_both(water_vapor):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (10,10), constrained_layout = True)

    im1 = axs[0].imshow(water_vapor)
    fig.colorbar(im1)
    axs[0].set_title("Atmospheric Water Vapor Levels (g/cm^2)")
    fig.show()

    im2 = axs[1].hist(water_vapor.flatten())
    axs[1].set_ylabel('Frequency')
    axs[1].set_xlabel('Water Vapor (g/cm^2)')
    axs[1].set_title('Atmospheric Water Vapor Histogram')
    fig.show()
    

def plot_h2o_map(water_vapor):
    plt.imshow(water_vapor)
    plt.colorbar()
    plt.title('Atmospheric Water Vapor (g/cm^2)')
    plt.show()

def plot_h2o_hist(water_vapor):
    plt.hist(water_vapor.flatten())
    plt.title('Water Vapor Distribution')
    plt.xlabel('Water Vapor (g/cm^2)')
    plt.ylabel('Frequency')
    plt.show()
    
import matplotlib.pyplot as plt
from spectral.io import envi
import numpy as np

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

def plot_h2o_compare(wv_test, wv_pred):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 6), constrained_layout = True)

    im1 = axs[0].imshow(wv_test)
    fig.colorbar(im1)
    im1.set_clim(0, 4.0)
    axs[0].set_title("WV Test (g/cm^2)")

    im2 = axs[1].imshow(wv_pred)
    fig.colorbar(im2)
    axs[1].set_title("WV Pred (g/cm^2)")
    im2.set_clim(0, 4.0)


def plot_masks_and_residuals(test, pred, test_paths, pred_paths idx, clim_1 = (0, 6), clim_2 = (-1, 1)):
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 4), constrained_layout = True)

    test = test[idx]
    pred = pred[idx]

    im1 = axs[0].imshow(test)
    fig.colorbar(im1)
    im1.set_clim(clim_1)
    axs[0].set_title("WV Test (g/cm^2)")
    
    im2 = axs[1].imshow(pred)
    fig.colorbar(im2)
    im2.set_clim(clim_1)
    axs[1].set_title("WV Pred (g/cm^2)")
    
    residuals = np.subtract(pred, test)
    im3 = axs[2].imshow(residuals, cmap = 'seismic')
    fig.colorbar(im3)
    im3.set_clim(clim_2)
    axs[2].set_title('Residuals (Pred - Test)');
    
    print(test_paths[idx])
    print(pred_paths[idx])
    
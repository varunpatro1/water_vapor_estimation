import netCDF4 as nc
import xarray as xr
from geoarray import GeoArray
from spectral.io import envi
import matplotlib.pyplot as plt
import numpy as np
from isofit.core.common import resample_spectrum
import os
import isofit
from isofit.core.sunposition import sunpos
import holoviews as hv
import hvplot.xarray
from datetime import datetime

def load_info(path_list):
    
    """
    Description: Loads the observation, radiance and mask header files. Extracts from these files
    radiance, atmospheric water vapor and zenith angle, packages them into a list and returns them
    File path order: obs, rad, mask

    """
    data = []
    for i in range(len(path_list)):
        data.append(envi.open(path_list[i]).open_memmap(interleave = 'bip'))
    rad = data[1]
    zen = np.average(data[0][:,:,4])
    #es_distance = data[0][:,:,10]
    water_vapor = data[2][:,:,6]

    data_list = []
    data_list.append(rad)

    data_list.append(zen)
    #data_list.append(es_distance)
    data_list.append(water_vapor)

    return data_list

def get_water_vapor(path_list):    
    return load_info(path_list)[2]
    

def get_irr(rad_path: str, irr_path: str):

    # Description: calculates irradiance based on kurudz irr file
    
    rad_header = envi.open(rad_path)

    wl = rad_header.metadata['wavelength']
    for i in range(len(wl)):
        wl[i] = float(wl[i])
    wl = np.array(wl)

    fwhm = rad_header.metadata['fwhm']
    for i in range(len(fwhm)):
        fwhm[i] = float(fwhm[i])
    fwhm = np.array(fwhm)

    # will need new irradiance file
    irr_file = os.path.join(
        os.path.dirname(isofit.__file__), "..", "data", "kurudz_0.1nm.dat") # same for anything TOA, pure solar irradiance

    irr_wl, irr = np.loadtxt(irr_file, comments="#").T
    irr = irr / 10  # convert to uW cm-2 sr-1 nm-1
    irr_resamp = resample_spectrum(irr, irr_wl, wl, fwhm)
    irr_resamp = np.array(irr_resamp, dtype=np.float32)
    irr = irr_resamp

    return irr

def calc_TOA_refl(rad, irr, zen):
    rho = (np.pi / np.cos(zen)) * rad / irr[np.newaxis, np.newaxis, :]
    return rho

def reshape_select(TOA_refls, water_vapor_vals, img_size: int, num_to_select: int):

    """
    Description: reshapes two arrays of TOA reflectances and water vapor values and randomly
    selects a user-prescribed number of pixels from each
    """
    TOA_refls_reshaped = TOA_refls.reshape((TOA_refls[0]*TOA_refls[1], TOA_refls[2]))
    water_vapor_vals_reshaped = water_vapor_vals.flatten()
    
    indices = np.random.randint(low = 0, high = img_size, size = num_to_select)
    subset_refls = TOA_refls_reshaped[indices]
    subset_water_vapor = water_vapor_vals_reshaped[indices]

    return subset_refls, subset_water_vapor





def main(file_path_list: list, irr_path: str):
    rad = load_info(file_path_list)[0]
    irr = get_irr(file_path_list[1], irr_path)
    zen = load_info(file_path_list)[1]

    TOA_refl = calc_TOA_refl(rad, irr, zen)
    return TOA_refl


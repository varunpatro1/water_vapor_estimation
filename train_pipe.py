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

def load_info(path_list: str):
    # order: obs, location, rad, mask
    data = []
    for i in range(len(path_list)):
        #print(path_list[i])
        data.append(envi.open(path_list[i]).open_memmap(interleave = 'bip'))

    """
    obs_header = envi.open(obs_path)
    location_header = envi.open(loc_path)
    rad_header = envi.open(rad_path)
    mask_header = envi.open(mask_path)

    obs_data = obs_header.open_memmap(interleave = 'bip')
    loc_data = location_header.open_memmap(interleave = 'bip')
    rad_data = rad_header.open_memmap(interleave = 'bip')
    mask_data = mask_header.open_memmap(interleave = 'bip')

    "

    lat = data[1][:,:,0]
    lon = data[1][:,:,1]
    elev = data[1][:,:,2

    """
    rad = data[2]
    zen = np.average(data[0][:,:,4])
    #es_distance = data[0][:,:,10]
    water_vapor = data[3][:,:,6]

    data_list = []
    data_list.append(rad)
    #data_list.append(lat)
    #data_list.append(lon)
    #data_list.append(elev)
    data_list.append(zen)
    #data_list.append(es_distance)
    data_list.append(water_vapor)

    return data_list

def get_irr(rad_path: str, irr_path: str):
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


def main(file_path_list: list, irr_path: str):
    rad = load_info(file_path_list)[0]
    irr = get_irr(file_path_list[2], irr_path)
    zen = load_info(file_path_list)[1]

    TOA_refl = calc_TOA_refl(rad, irr, zen)
    return TOA_refl

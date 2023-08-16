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

def extract_from_header(path_list, irr_path):
    
    """
    Description: Loads the observation, radiance and mask header files. Extracts from these files
    radiance, atmospheric water vapor and zenith angle, packages them into a list and returns them
    
    File path order: obs, rad, mask

    """
    
    data = []
    for i in range(len(path_list)):
        data.append(envi.open(path_list[i]).open_memmap(interleave = 'bip'))
    rad = data[1]
    zen = zen = np.deg2rad(np.average(data[0][:,:,4]))
    es_distance = data[0][:,:,10]
    water_vapor = data[2][:,:,6]

    irr = calc_irr(path_list[1], irr_path)
    irr = irr * ((np.average(es_distance))**2)

    return rad, zen, irr, water_vapor

def extract_subset_from_header(path_list, irr_path):

    #TODO - update documentation
    """
    Description: Loads the observation, radiance and mask header files. Extracts from these files
    radiance, atmospheric water vapor and zenith angle, packages them into a list and returns them

    File path order: obs, rad, mask

    """

    subset = ([640],np.arange(1242,dtype=int).tolist())
    data = []
    for i in range(len(path_list)):
        data.append(envi.open(path_list[i]).open_memmap(interleave = 'bip')[subset])
        print(data[i].shape)
    rad = data[1]
    zen = zen = np.deg2rad(np.average(data[0][...,4]))
    es_distance = data[0][...,10]
    water_vapor = data[2][...,6]

    irr = calc_irr(path_list[1], irr_path)
    irr = irr * ((np.average(es_distance))**2)

    return rad, zen, irr, water_vapor


def calc_irr(rad_path: str, irr_path: str):
    
    rad_header = envi.open(rad_path)

    wl = rad_header.metadata['wavelength']
    fwhm = rad_header.metadata['fwhm']
    for i in range(len(wl)):
        wl[i] = float(wl[i])
        fwhm[i] = float(fwhm[i])
    wl = np.array(wl)
    fwhm = np.array(fwhm)

    irr_path = '../irr.npy'
    data = np.load(irr_path)
    irr_wl = data[:,0]
    irr = data[:,1]

    irr_resamp = resample_spectrum(irr, irr_wl, wl, fwhm)
    irr_resamp = np.array(irr_resamp, dtype=np.float32)
    irr = irr_resamp

    return irr

def transform_entire_scene(file_path_list, irr_path):

    # perform TOA reflectance calculation
    rad, zen, irr, wv = extract_subset_from_header(file_path_list, irr_path)
    refl = (np.pi / np.cos(zen)) * (rad / irr[np.newaxis, np.newaxis, :])

    # reshape and randomly select
    img_size = refl.shape[0]*refl.shape[1]
    refl = refl.reshape((refl.shape[0]*refl.shape[1], refl.shape[2]))
    wv = wv.flatten()

    return refl, wv

def transform_and_select(file_path_list, irr_path, num_to_select):

    # perform TOA reflectance calculation
    rad, zen, irr, wv = extract_from_header(file_path_list, irr_path)
    refl = (np.pi / np.cos(zen)) * (rad / irr[np.newaxis, np.newaxis, :])

    # reshape and randomly select
    img_size = refl.shape[0]*refl.shape[1]
    indices = np.random.randint(low = 0, high = img_size, size = num_to_select)
    refl = refl.reshape((refl.shape[0]*refl.shape[1], refl.shape[2]))
    wv = wv.flatten()
    refl = refl[indices]
    wv = wv[indices]

    return refl, wv

def convert_rad(file_path_list, irr_path):
    
    # perform TOA reflectance calculation
    rad, zen, irr, wv = extract_from_header(file_path_list, irr_path)
    refl = (np.pi / np.cos(zen)) * (rad / irr[np.newaxis, np.newaxis, :])

    return refl

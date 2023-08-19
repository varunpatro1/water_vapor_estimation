import datetime
import argparse
import glob
from spectral.io import envi
import numpy as np
import subprocess
import os
import pandas as pd
import ray
from preprocess import *


def get_rdn(fid):
    rdn_file = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/*_l1b_rdn_*.hdr'))[-1]
    return rdn_file

def get_obs(fid):
    obs_file = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/*_l1b_obs_*.hdr'))[-1]
    return obs_file

def get_mask(fid):
    mask_file = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l2a/*_l2a_mask_*.hdr'))[-1]
    return mask_file


@ray.remote
def get_single_scene_data(rdn_file,obs_file,mask_file,irr_path,fid_index, fid):
    # perform TOA reflectance calculation
    rad, zen, irr, wv, mask_dict = extract_subset_from_header([obs_file,rdn_file,mask_file], irr_path)
    refl = (np.pi / np.cos(zen)) * (rad / irr[np.newaxis, np.newaxis, :])

    # reshape and randomly select
    img_size = refl.shape[0]*refl.shape[1]
    refl = refl.reshape((refl.shape[0]*refl.shape[1], refl.shape[2]))
    wv = wv.flatten()

    out_fid = np.ones(len(wv),dtype=int)*fid_index
    fid = np.ones(len(wv),dtype=int)*fid

    return refl, wv, out_fid, fid, mask_dict


def main():

    parser = argparse.ArgumentParser(description="Run coverage vector tiling") #TODO
    parser.add_argument('fid_list', type=str) #TODO
    parser.add_argument('--irr_file', type=str, default='irr.npy') #TODO
    parser.add_argument('--output_file', type=str, default='outdir/munged') #TODO
    args = parser.parse_args()

    fids = open(args.fid_list,'r').readlines()
    fids = [x.strip() for x in fids]

    rdn_files = [get_rdn(x) for x in fids]
    obs_files = [x.replace('_rdn_','_obs_') for x in rdn_files]
    mask_files = [x.replace('_rdn_','_mask_').replace('l1b','l2a') for x in rdn_files]

    subset = [os.path.isfile(x) and os.path.isfile(y) and os.path.isfile(z) for x,y,z in zip(rdn_files,obs_files,mask_files)]

    rdn_files = np.array(rdn_files)[np.array(subset)].tolist()
    obs_files = np.array(obs_files)[np.array(subset)].tolist()
    mask_files = np.array(mask_files)[np.array(subset)].tolist()

    ray.init()
    jobs = []
    for _fid, fid in enumerate(fids):
        jobs.append(get_single_scene_data.remote(rdn_files[_fid], obs_files[_fid], mask_files[_fid], args.irr_file,_fid, fid))

    output_rfl, output_wv, output_idx = [],[],[], []
    rreturn = [ray.get(jid) for jid in jobs]
    for _res, res in enumerate(rreturn):
        output_rfl.append(res[0])
        output_wv.extend(res[1].tolist())
        output_idx.extend(res[2].tolist())
        output_fid.extend(res[3].tolist())

    output_rfl = np.vstack(output_rfl)
    output_wv = np.array(output_wv)
    output_idx = np.array(output_idx)
    output_fid = np.array(output_fid)
    print(output_rfl.shape)
    print(output_wv.shape)
    print(output_idx.shape)
    print(output_fid.shape)
    
    dict = {'refls: ': output_rfl, 'water vapor': output_wv, 'fid index': output_idx, 'fid': output_fid}


if __name__ == "__main__":
    main()

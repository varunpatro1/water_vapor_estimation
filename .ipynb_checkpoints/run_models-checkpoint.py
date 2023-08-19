from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def load_and_remove(data):
    
    refls = data['output_rfl']
    wv = data['output_wv']
    fids = data['output_idx']
    
    values, counts = np.unique(fids, return_counts=True)
    scenes = np.split(refls, 743)
    wv_by_scene = np.split(wv, 743)
    scenes = np.array(scenes)
    wv_by_scene = np.array(wv_by_scene)
    
    bad = []
    for i in range(wv_by_scene.shape[0]):
        scene = wv_by_scene[i].copy()
        if scene[np.where((scene > 6) | (scene < 0))].shape[0] != 0:
            bad.append(i)
    
    wv_by_scene = np.delete(wv_by_scene, bad, axis = 0)
    scenes = np.delete(scenes, bad, axis = 0)

    return scenes, wv_by_scene

def split_data(scenes, wv_by_scene):
    
    np.random.seed(42)
    X_idx = np.random.choice(718, 300)
    y_idx = X_idx
    X_train_idx, X_test_idx, y_train_idx, y_test_idx = train_test_split(X_idx, y_idx, test_size = 0.2, random_state = 50)
    
    X_train = scenes[X_train_idx, :, :]
    X_train = X_train.reshape((X_train.shape[0]*X_train.shape[1], X_train.shape[2]))
    X_test = scenes[X_test_idx, :, :]
    X_test = X_test.reshape((X_test.shape[0]*X_test.shape[1], X_test.shape[2]))
    y_train = wv_by_scene[y_train_idx, :]
    y_train = y_train.flatten()
    y_test = wv_by_scene[y_test_idx, :]
    y_test = y_test.flatten()

    return X_train, y_train, X_test, y_test

def run_rf(X_train, y_train, X_test, y_test):
        
    rf = RandomForestRegressor(random_state = 0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    outpath = 'model_outdir/300_rf_6'
    np.savez(outpath, y_pred, y_test, y_train)

def run_dt(X_train, y_train, X_test, y_test):

    dt = DecisionTreeRegressor(random_state = 0)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    outpath = 'model_outdir/300_dt_6'
    np.savez(outpath, y_pred, y_test, y_train)


def main():
    data = np.load('../train_no_clouds.npz')
    scenes, wv_by_scene = load_and_remove(data)
    X_train, y_train, X_test, y_test = split_data(scenes, wv_by_scene)
    run_rf(X_train, y_train, X_test, y_test)
    

if __name__ == "__main__":
    main()
    

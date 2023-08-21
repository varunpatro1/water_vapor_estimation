from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


def load_and_remove(data, wv_upper_bound, shortened_bands):
    
    refls = data['output_rfl']
    wv = data['output_wv']
    fids = data['output_idx']
    
    values, counts = np.unique(fids, return_counts=True)
    scenes = np.split(refls, 743)
    wv_by_scene = np.split(wv, 743)
    scenes = np.array(scenes)
    wv_by_scene = np.array(wv_by_scene)
    
    if shortened_bands:
        scenes = scenes[:,:,60:100]

    bad = []
    for i in range(wv_by_scene.shape[0]):
        scene = wv_by_scene[i].copy()
        if scene[np.where((scene > wv_upper_bound) | (scene < 0))].shape[0] != 0:
            bad.append(i)
            
    wv_by_scene = np.delete(wv_by_scene, bad, axis = 0)
    scenes = np.delete(scenes, bad, axis = 0)

    return scenes, wv_by_scene

def split_data(scenes, wv_by_scene, per_pixel, all_samples, num_scenes):

    if per_pixel:
        scenes = scenes.reshape((scenes.shape[0]*scenes.shape[1], scenes.shape[2]))
        wv_by_scene = wv_by_scene.flatten()
        X_train, X_test, y_train, y_test = train_test_split(scenes, wv_by_scene, test_size = 0.2, random_state = 50)

    else:
        # per scene
        if all_samples:
            X_idx = np.arange(718)
        else:
            np.random.seed(42)
            X_idx = np.random.choice(718, num_scenes)

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

def run_rf(X_train, y_train, X_test, y_test, file_outpath):
    
    rf = RandomForestRegressor(random_state = 0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    outpath = file_outpath
    np.savez(outpath, y_pred, y_test, y_train)

def run_dt(X_train, y_train, X_test, y_test, file_outpath):

    dt = DecisionTreeRegressor(random_state = 0)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    outpath = file_outpath
    np.savez(outpath, y_pred, y_test, y_train)

def run_mlpreg(X_train, y_train, X_test, y_test, file_outpath, per_pixel):

    if per_pixel:
        clf = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(50, 50), activation = 'tanh', random_state=1)
    else:
        clf = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(300,300), activation = 'tanh', random_state=1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    outpath = file_outpath
    np.savez(outpath, y_pred, y_test, y_train)


def main():
    data = np.load('train_no_clouds.npz')
    scenes, wv_by_scene = load_and_remove(data, wv_upper_bound = 5, shortened_bands = False)
    X_train, y_train, X_test, y_test = split_data(scenes, wv_by_scene, per_pixel = True, all_samples = True, num_scenes = 718)
    run_dt(X_train, y_train, X_test, y_test, 'model_outdir/per_pixel/all_dt_5')
    run_rf(X_train, y_train, X_test, y_test, 'model_outdir/per_pixel/all_rf_5')
    run_mlpreg(X_train, y_train, X_test, y_test, 'model_outdir/per_pixel/all_mlpreg_5', per_pixel = True)

if __name__ == "__main__":
    main()

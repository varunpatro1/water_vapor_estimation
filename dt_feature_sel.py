import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text
import pandas as pd
import random
from sklearn import tree
import joblib
se
def load_and_remove(data, wv_upper_bound, num_bands):
    
    refls = data['output_rfl']
    wv = data['output_wv']
    fids = data['output_idx']
    
    values, counts = np.unique(fids, return_counts=True)
    scenes = np.split(refls, 743)
    wv_by_scene = np.split(wv, 743)
    scenes = np.array(scenes)
    wv_by_scene = np.array(wv_by_scene)
    
    if num_bands == 40:
        scenes = scenes[:,:,60:100]

    bad = []
    for i in range(wv_by_scene.shape[0]):
        scene = wv_by_scene[i].copy()
        if scene[np.where((scene > wv_upper_bound) | (scene < 0))].shape[0] != 0:
            bad.append(i)
            
    wv_by_scene = np.delete(wv_by_scene, bad, axis = 0)
    scenes = np.delete(scenes, bad, axis = 0)

    return scenes, wv_by_scene


def split_data(scenes, wv_by_scene, per_pixel, dataset_fraction, num_scenes_to_select, random_state):

    rs = random_state
    if per_pixel:
        scenes = scenes.reshape((scenes.shape[0]*scenes.shape[1], scenes.shape[2]))
        wv_by_scene = wv_by_scene.flatten()
        X_train, X_test, y_train, y_test = train_test_split(scenes, wv_by_scene, test_size = 0.2, random_state = rs)

    else:
        # per scene
        X_train, X_test, y_train, y_test = train_test_split(scenes, wv_by_scene, test_size = 0.2, random_state = rs)
        X_train = X_train.reshape((X_train.shape[0]*X_train.shape[1], X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0]*X_test.shape[1], X_test.shape[2]))
        y_train = y_train.flatten()
        y_test = y_test.flatten()

    return X_train, y_train, X_test, y_test

def run_dt(X_train, y_train, X_test, y_test, file_outpath, wl_file, num_bands, iter):

    if num_bands == 285:
        wl = np.loadtxt(wl_file).tolist()
        band_number = np.arange(285)
        feature_names = [f'Band {i+1}: {j} nm' for i,j in zip(band_number, wl)]

    else:
        wl = np.loadtxt(wl_file).tolist()[59:99]
        band_number = np.arange(59,99)
        feature_names = [f'Band {i+1}: {j} nm' for i,j in zip(band_number, wl)]
        
    dt = DecisionTreeRegressor(random_state = 0)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    np.savez(f'{file_outpath}_pred_{iter}', y_pred, y_test, y_train)
    joblib.dump(dt, f'{file_outpath}_model_{iter}')

    text_representation = tree.export_text(dt, feature_names = feature_names)
    text_file = open(f'{file_outpath}_txt_rep_{iter}', "w")
    text_file.write(text_representation)
    text_file.close()

    most_features_frame = pd.DataFrame(
    data = dt.feature_importances_,
    columns= ['importance'],
    index = np.arange(num_bands),
).sort_values(by=["importance"], ascending=False)

    feature_order = most_features_frame.index
    top_feature_list = [i for i in feature_order]
    features = np.array(feature_names)[np.array(top_feature_list)]
    importances = -np.sort(-dt.feature_importances_)
    dat = np.array([features, importances])
    dat = dat.T
    np.savetxt(f'{file_outpath}_features_{iter}', dat, fmt="%s", delimiter = ', Importance: ')

def main():
    # LOAD DATA
    data = np.load('../train_no_clouds.npz')

    # SET PARAMETER VALUES
    wv_upper_bound = [6]
    num_bands = [40]
    dataset_frac = 'all'
    
    pixel_based = False
    scenes_to_select = 300
    random_states = random.sample(range(10, 1000), 3)

	
    # MODIFY DATA AND RUN MODELS
    count = 0
    if dataset_frac == 'all':
        for state in random_states:
            for bands in num_bands:
                for wv in wv_upper_bound:
                    scenes, wv_by_scene = load_and_remove(data, wv_upper_bound = wv, num_bands = bands)
                    X_train, y_train, X_test, y_test = split_data(scenes, wv_by_scene, per_pixel = pixel_based, \
                                                                      dataset_fraction = dataset_frac, num_scenes_to_select = scenes_to_select, random_state = state)
                    run_dt(X_train, y_train, X_test, y_test, file_outpath = f'../dt_feat_sel/first_run/random_tts_states/all_dt_{wv}_{bands}', \
                           wl_file = '../wavelength.txt', num_bands = bands, iter = count)
            count+=1

if __name__ == "__main__":
    main()

import numpy as np
import preprocess
import time

def convert_scenes(scene_paths, irr_path)

TOA_refls = []
water_vapor_vals = []
num_scenes = len(scene_paths)

start_time = time.time()
for i in range(num_scenes):
    TOA_refl, water_vapor = preprocess.transform_entire_scene(scene_paths[i], irr_path)
    TOA_refls.append(TOA_refl)
    water_vapor_vals.append(water_vapor)

    if (i+1) % 10 == 0:
        print('scene: ', i+1)
    
end_time = time.time()
elapsed = end_time - start_time
print('Executed in: ', elapsed, ' seconds')
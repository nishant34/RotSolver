import json
import numpy as np

file = open(r"E:/Nerf_ablation/open_mvg_results_statue/reconstruction_global/sfm_data_extrinsics.json")
data = json.load(file)

rotation_list = []
translation_list = []

for curr_details in data["extrinsics"]:
    rotation_list.append(curr_details['value']['rotation'])
    translation_list.append(curr_details['value']['center'])

rotation_list = np.array(rotation_list)
translation_list = np.array(translation_list)
print(rotation_list.shape)
print(translation_list.shape)
np.save("E:/Nerf_ablation/open_mvg_results_statue/reconstruction_global/rotations.npy", rotation_list)
np.save("E:/Nerf_ablation/open_mvg_results_statue/reconstruction_global/translations.npy", translation_list)


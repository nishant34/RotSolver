import numpy as np
import os


class relative_camera_poses_data:
    
    """
    Class to load relative rotations data for differentiable rotation averaging.
    THe data format should be as follows-->
    Root_dir
    | -rotations.npy --> relative rotations pairwise.
    | -translations.npy --> relative translations pair wise.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.rot_file_path = os.path.join(data_dir, "rotations.npy")
        self.trans_file_path = os.path.join(data_dir, "translations.npy")
        
        #extracting data
        self.relative_rotations = np.load(self.rot_file_path)
        self.relative_translations = np.load(self.trans_file_path)
        #reshaping from (num_images,3,3) to (num_images,9)
        self.relative_rotations = np.reshape(self.relative_rotations, (-1,9))
        
        assert self.relative_rotations.shape[0] == self.relative_translations.shape[0], "Inconsistent data"

    
    def _get_rot_(self, index):
        return self.relative_rotations[index]

    
    def _get_trans_(self, index):
        return self.relative_translations[index]
    

    def _get_all_rots_(self):
        return self.relative_rotations
    
    def _get_all_trans_(self):
        return self.relative_translations
    
    def _get_num_features_(self):
        return self.relative_rotations.shape[-1]
    
    def _len_(self):
        return self.relative_rotations.shape[0]
    


    

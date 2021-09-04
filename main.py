from rot_average import NeuralRotAvg
from utils import *
import numpy as np 
#import jax 
#from rot_average import *
from rotation_dataloader import *
from gnns import RotSolverModel
from common import *


if __name__=="__main__":

 #defining the dataloader.................
 rot_loader = relative_camera_poses_data(data_dir)
 #testing.....
 relative_rots = rot_loader._get_all_rots_()
 relative_rots = relative_rots[:-1] #temporary
 #print(relative_rots.shape) #(num_examples,9)

 print("The number of Epochs for training:{}".format(Num_Epochs))
 print("The initial learning rate is:{}".format(lr))
 print("The number of images in the set are:{}".format(Num_Images))
 print("The input fetaure shape is:{}".format(relative_rots.shape[-1]))
 
 #creating the self-supervised-gt
 target = create_self_supervised_gt(Num_Images)

 #generating the adj matrix
 adj = get_full_connected_adj_matrix(Num_Images)

 #defining the graph neural network based rot solver model......
 rot_solver = RotSolverModel(relative_rots.shape[-1], lr)
 #rot_solver._initilialize_model_(Hidden_features, relative_rots.shape[-1])

 #defining the wrapper model...........
 model = NeuralRotAvg()
 model._set_rot_solver_model_(rot_solver)
 model._set_dataloader_(rot_loader)
 model._set_adj_(adj)
 model._initialize_training_model_(Hidden_features)
 rot_solver._get_optim_()
 

 #defining the training loop:
 print("The training has been started........")
 model.train(Num_Epochs)


 



 

 




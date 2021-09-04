import numpy as np 
# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
import abc 
import datetime
import time 
from gnns import *
from tqdm import tqdm
import jax
from common import *

"abstract functions based class for a differentiable rotation averaging function"
class RotAveraging:
    def __init__(self):
        """
        The attributes in rotation averaging are defined as follows:
        1.) num_nodes:  number of images in the image set-->N
        2.) global rotations : rotation matrixes representation of camera poses for each of the images --> (N,3,3)
        3.) indices: consists of pairs (i,j) for which we have relative rotations availbale initially --> (M,2)
        4.) relative rotations: for each of the pair of images in indices, this contains the relative rotations RjRi.T -->(M,3,3) 
        """
        self.num_nodes = 5    #by default 5 nodes are assumed   
        self.global_rots = np.zeros((self.num_nodes,3,3))  #(num_images,3,3)--> shape for rot at for each image:(3,3)
        self.num_edges = 10 #by default we have 7 pairs of iamges for which reative rotations are available initially --> M=7
        self.indices = np.zeros((self.num_edges,2)) #pairs defining the edges --> (M,2)
        self.relative_rots = np.zeros((self.num_edges,3,3)) #relative rots for each pair corresponding to an edge --> (M,3,3)
        self.log_path = None

    def  __get_loss__(self, relative_rots) :
       """
       wrapper funciton for getting the objective to update the rot_averaging angles
       """
       return self._get_loss(relative_rots)
    
    def __get_output__(self):
        """
        wrapper function to output the global rotations based on, for e.g. Neural Networks, given the relative rotation values
        along wiht the remaining attributes.
        """
        return self._get_output_rots()

    def __get_status__(self):
        "get the current attribute values printed, involved in rotation averging"
        print("The number of nodes in grpahical model of rot averaging:{}".format(self.num_nodes))
        print("The number of edges in graphical model of rotation averaging:{}".format(self.num_edges))
        if self.log_path is not None:
            np.save(self.log_path+str(datetime.datetime.now())+".npy",self.global_rots)
            print("The current global rotation values have been saved in the npy file format.")
    
    def __reset_attributes__(self, num_nodes=None, num_edges=None, relative_rots=None, global_rots=None):
        "resetting the attributes to some given value or to the defualt values."
        if self.num_nodes is not None:
            self.num_nodes = num_nodes
        else: 
            self.num_nodes=5  #going to the default case

        if self.num_edges is not None:
            self.num_edges = num_edges
        else: 
            self.num_edges=7  #going to the default case

        if self.relative_rots is not None:
            self.relative_rots = relative_rots
        else: 
            self.relative_rots = np.zeros((self.num_edges,3,3))  #going to the default case
        
        if self.global_rots is not None:
            self.global_rots = global_rots
        else: 
            self.global_rots = np.zeros((self.num_nodes,3,3))  #going to the default case
                
                
    @abc.abstractmethod
    def _get_loss(self, relative_rots):
        "abstract function to get the rot averaging loss for a given relative rotation set."
        return 
    
    @abc.abstractmethod
    def _get_output_rots(self):
        """abstract method to generate the output values of global rotations using 
           linear/non-linear dependency(eg. Neural networks) given relative rotations."""
        
        return 


class NeuralRotAvg(RotAveraging):
    """
    A wrapper class implementing the non-linear optimization(Adam optimization algorithm) algorithm for rotation averaging using neural networks.
    Implementation is based on graph neural networks.
    """
    def __init__(self):
        "initiliazer for neural networks based rot averaging."
        super(NeuralRotAvg, self).__init__()
        self.rng_key = jax.random.PRNGKey(seed) #to be decided.....

    def _set_rot_solver_model_(self, rot_solver):
        "setting up the model that will be responsible for generating the output"
        self.rot_solver = rot_solver
    
    def _set_dataloader_(self, dataloader):
        "Setting up the customized dataloader, so that we can run this on the type of data we want"
        self.dataloader = dataloader
    
    def _set_adj_(self, adj):
        self.adj = adj
    
    def _initialize_training_model_(self, n_hidden, dropout=0.5):
        "Initiializing the training model, which here is the 2 layered-GCN......"
        self.n_out = self.dataloader._get_num_features_()
        #self.rot_solver_init, self.rot_solver_pred_fun = self.rot_solver._graph_conv_network_(n_hidden, n_out, dropout)
        self.rot_solver._initialize_model_(n_hidden, 9)
        input_shape = (1, self.dataloader._len_(), self.dataloader._get_num_features_())
        self.rng_key, init_key = random.split(self.rng_key)
        _, self.rot_solver.init_params = self.rot_solver._init_(init_key, input_shape)
 
    def train_step(self, epoch, show_time=False):
        "Updating parameters using all the rotations at each step......"
        start_time = time.time()
        rot_data = self.dataloader._get_all_rots_()
        #data to be used, here complete data in 1 batch.....
        batch = (self.dataloader._get_all_rots_(), self.adj, True, self.rng_key) #idx_train--> check 
        #training update 
        self.rot_solver.opt_state = self.rot_solver.update(epoch, self.rot_solver.opt_state, batch)
        epoch_time = time.time() - start_time

        #calculating the loss....
        self.rot_solver.params = self.rot_solver.get_params(self.rot_solver.opt_state)
        loss = self.rot_solver.loss(self.rot_solver.params, batch)
        
        # new random key....
        self.rng_key, _ = random.split(self.rng_key)

        if show_time:
            return loss, epoch_time
        
        return loss
    
    def train(self, num_epochs, show_time=False):
        print("Number of training epochs:{}".format(num_epochs))
        print("Number of nodes:{}".format(self.dataloader._len_()))
        print("Input feature dimension:{}".format(self.dataloader._get_num_features_()))
        print("Output_fetaure_dimension:{}".format(self.dataloader._get_num_features_()))
        print("Started training for differentiable rotation averaging........")

        #training......
        for epoch in tqdm(range(num_epochs)):
            loss = self.train_step(epoch)
            print("--------------------------------")
            print("Epoch:{} Loss:{}".format(epoch, loss))
        
        print("The training has been completed.")




        




    


    
    

        
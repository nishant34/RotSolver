#from os import pardir
import numpy.random as npr
import jax.numpy as np 
import jax.random as random
import jax.nn as nn
from jax import grad 
from jax.nn.initializers import glorot_uniform
from jax.experimental.stax import Dropout
from jax.experimental import optimizers
import math
import jax


"This file has gnns implemented(for differentiable rotation averaging) in jax to be merged with mip-nerf"

Num_curr = 18

class RotSolverModel:
    "class implementing  graph convolutional networks........"
    def __init__(self, input_shape, lr=0.001):
        self.input_shape  = input_shape
        self.lr = lr

    def _graph_convolution_(self, out_features, bias=False):
        "layer constructor for gcns in jax......."

        def init_fun(rng, input_shape):
            output_shape = input_shape[:-1] + (out_features,)
            k1, k2 = random.split(rng)
            W_init, b_init = glorot_uniform(), np.zeros
            W = W_init(k1, (input_shape[-1], out_features))
            b = b_init(k2, (out_features,)) if bias else None
            return output_shape, (W, b)
        
        def apply_fun(params, x, adj, **kwargs):
            W, b = params
            support  = np.dot(x,W)
            out =support
            # out = np.matmul(adj, support)
            # if bias:
            #     out+=b
            return out
        
        return init_fun, apply_fun
    
    def _graph_conv_network_(self, nhid, nout=9, dropout=0.5):
        "implementing 2-layered GCN"
        gc1_init, gc1_fun = self._graph_convolution_(nhid)
        _, drop_fun = Dropout(dropout)
        gc2_init, gc2_fun = self._graph_convolution_(nout)

        init_funs = [gc1_init, gc2_init]

        def init_fun(rng, input_shape):
            params = []
            for init_fun in init_funs:
                rng, layer_rng = random.split(rng)
                input_shape, param = init_fun(layer_rng, input_shape)
                params.append(param)
            
            return input_shape, params
        

        def apply_fun(params, X, adj, is_training=False, **kwargs):
            rng = kwargs.pop('rng', None)
            k1, k2, k3, k4 = random.split(rng,4)
            # print("Inside the apply function, the input shape is:{}".format(X.shape))
            # print("Inside the apply function, the input is:{}".format(X))
            x = drop_fun(None, X, is_training=is_training, rng=k1)
            #print("after 1st layer:{}".format(x))
            x = gc1_fun(params[0], x, adj, rng=k2)
            #print("after 2nd layer:{}".format(x))
            x = nn.relu(x)
            #print("after 3rd layer:{}".format(x))
            x = drop_fun(None, x, is_training=is_training, rng=k3)
            #print("after 4th layer:{}".format(x))
            x = gc2_fun(params[1], x, adj, rng=k4)
            #print("after 5th layer:{}".format(x))
            #x = nn.log_softmax(x)
            #print("after 6th layer:{}".format(x))
            return x
        
        return init_fun, apply_fun
    
    def _initialize_model_(self, n_hidden, n_out, dropout=0.5):
        self._init_, self._pred_fun_ = self._graph_conv_network_(n_hidden, n_out, dropout)
        
    
    def update(self, iter, opt_state, batch):
        params = self.get_params(opt_state)
        return self.opt_update(iter, grad(self.loss)(params, batch), opt_state)
    
    def loss(self, params, batch):
        """
        The idxes of the batch indicate which nodes are used to compute the loss.
        """
        inputs, adj, is_training, rng = batch #idx--> train_check
        inputs_1 = np.reshape(inputs, (-1,3,3))
        inputs_1 = np.repeat(inputs_1, Num_curr, axis=0)
        #print("inputs_1 shape is:{}".format(inputs_1.shape))
        preds = self._pred_fun_(params, inputs, adj, is_training=is_training, rng=rng)
        preds = np.reshape(preds, (-1,3,3))
        self.preds = preds
        #print("The preds are:{}".format(preds))
        d = np.outer(preds, np.linalg.inv(preds)) #(Num_curr*9,Num_curr*9)
        #d = np.reshape(d,(-1,9))
        d = np.reshape(d,(Num_curr*9,Num_curr,9))
        d = np.sum(d,-1)
        d = np.reshape(d,(Num_curr,9,Num_curr))
        d = np.transpose(d,(0,2,1))
        d = np.reshape(d,(-1,9))
        #e =np.tril_indices(d)
        #out_rel = d[e] #(55,9)
        out_rel = d
        out_rel = np.reshape(out_rel, (-1,3,3))
        #print("out rel is:{}".format(out_rel))
        loss = out_rel@np.linalg.inv(inputs_1)
        #print("the loss before log is:{}".format(loss))
        #loss = np.log(loss)
        loss = loss/math.sqrt(2)
        loss = Forbenius_norm(loss)
        loss = np.sum(loss)
        return loss


    def _get_optim_(self):
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(self.lr)
        self.opt_state = self.opt_init(self.init_params)



def Forbenius_norm(x):
    """
    Function to calculate the Frobenius Norm for any matrix, defined as square root of the sum of absolute square of its elements.
    """
    return jax.numpy.linalg.norm(x,ord="fro",axis=(1,2))


# def relu(x):
#     return np.maximum(0,x)


# relu_grad = grad(relu)

# def linear(params, x):
#     w,b = params
#     return w*x+b

# linear_grad = grad(linear)
# w,b = 4,4
# value_grad_linear = linear_grad((w,b),2)

# def loss(params, dataset):
#     x, y = dataset
#     pred = linear(params, x)
#     return np.square(pred-y).mean()

# loss_grad =grad(loss)
# params_grad = ((w,b),(2,5))

    




   






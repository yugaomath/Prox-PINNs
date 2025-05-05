###############################################################################

import numpy as np 
import torch 
from torch.autograd import grad
from Utils.GenData import GenData

###############################################################################

class Problem():

    def __init__(self, **args):
        '''
        The definition of the Problem
        '''
        self._name = 'Problem_Module'

    
    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        raise NotImplementedError

    @property
    def lb(self):
        raise NotImplementedError

    @property
    def ub(self):
        raise NotImplementedError


    def _grad_u(self, x:torch.tensor, u:torch.tensor)->torch.tensor:
        '''
        Input:  
            x: size(?,d)
            u: size(?,1)
        Output: 
            du: size(?,d)
        '''
        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        return du
    
    ###########################################################################

    def _div_u(self, x:torch.tensor, du:torch.tensor)->torch.tensor:
        '''
        Input: 
            x_list: [ size(?,1) ]*d
            du: size(?,d)
        Output: 
            Lu: size(?,1)
        '''
        #
        Lu = torch.zeros_like(du[:,0:1])
        for d in range(x.shape[1]):
            Lu += grad(inputs=x, outputs=du[:,d:d+1], 
                       grad_outputs=torch.ones_like(du[:,d:d+1]), 
                       create_graph=True)[0][:,d:d+1]
        return Lu
    
    ###########################################################################
    
    def get_observe(self, Nx_in:int=None, Nx_bd_each_side:int=None)->torch.tensor:
        '''
        Get the observation
            Nx_in: size of inside sensors
            Nx_bd_each_side: size of sensors on each boundary
                or 
               None (means loading from saved file.)
        Output:
            data_observe: dict={}
        '''
        raise NotImplementedError
    
    def get_test(self, x:torch.tensor=None)->torch.tensor:
        '''
        Input:
            x: size(?,d) 
                or
            None
        Output:
            u: size(?,1)
                or
            u: size(?,1)
            x: size(?,1)
        '''
        raise NotImplementedError
    
    def fun_bd(self, model:torch.nn.Module, x_list:list)->torch.tensor:
        '''
        The boundary conditions
        Input: 
            model:
            x_list: list= [size(n,1)]*2d    
        Output:  
            cond: size(n*2d*?,1)
        '''
        raise NotImplementedError

    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        Input: 
            x: size(?,d)
        Output: 
            f: size(?,1)
        '''
        raise NotImplementedError

    ########################################################################### 
    
    def get_point_cube(self, GenData:GenData, N_xin: int, N_xbd_each_face: int=None) -> dict:
        '''
        Input:
            GenData:
            N_xin: the number of points in the domain
            N_xbd_each_face: the number of points on the boundary (each face)
        Output:
            data_point: {'x_in', 'x_bd_list'}
        '''
        data_point = {}
        #
        x_in = GenData.get_in(Nx_size = N_xin)
        data_point['x_in'] = x_in 
        
        #
        if N_xbd_each_face is not None:
            x_bd_list = GenData.get_bd(N_bd_each_face = N_xbd_each_face)
            data_point['x_bd_list'] = x_bd_list

        return data_point
        
    ###########################################################################
    
    def get_point_disk(self, N_xin: int, N_xbd_each_face: int=None) -> dict:
        '''
        Input:
            GenData:
            N_xin: the number of points in the domain
            N_xbd_each_face: the number of points on the boundary (each face)
        Output:
            data_point: {'x_in', 'x_bd_list'}
        '''
        data_point = {}
        #
        r = self._R * np.sqrt(np.random.rand(N_xin, 1))  # r = R * sqrt(U)
        theta = 2 * np.pi * np.random.rand(N_xin, 1)     # theta in [0, 2*pi)
        
        x_in = np.concatenate([r * np.cos(theta), r * np.sin(theta)], axis=1)
        data_point['x_in'] = torch.from_numpy(x_in.astype(self._numpy_dtype))
        
        th = 2 * np.pi * np.random.rand(N_xbd_each_face, 1)   # theta in [0, 2*pi)
        x_bd = np.concatenate([self._R * np.cos(th), self._R * np.sin(th)], axis=1)
        
        x_bd_list = []
        x_bd_list.append(torch.from_numpy(x_bd.astype(self._numpy_dtype)))
        data_point['x_bd_list'] = x_bd_list

        return data_point
    
    ###########################################################################
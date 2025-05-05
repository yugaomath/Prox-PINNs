###############################################################################
import math
import torch 
import numpy as np 
from torch.autograd import Variable
###############################################################################
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
###############################################################################
from Utils.GenData import GenData
import Problems.Module as Module
###############################################################################

class Problem(Module.Problem):

    def __init__(self, **args):
        '''
        Input:
            np_type: np.float
            torch_type: torch.float
        '''
        #######################################################################
        
        self._name = args['problem_name']
        self._dim = args['problem_dim']
        self._domain = args['problem_domain']
        self._eta = 1e-3
        self._N_test = args['problem_N_test']

        self._lb = np.array(args['problem_lb'])
        self._ub = np.array(args['problem_ub'])
        
        self._numpy_dtype = args['numpy_dtype']
        self._torch_dtype = args['torch_dtype']
        
        #######################################################################
        
        self._gen_data = GenData(d=self._dim, x_lb=self._lb, x_ub=self._ub, dtype = self._numpy_dtype)
        
        #######################################################################
    
    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return self._dim
    
    @property
    def lb(self):
        lb = self._lb.reshape(-1, self.dim)
        return torch.from_numpy(lb.astype(self._numpy_dtype))

    @property
    def ub(self):
        ub = self._ub.reshape(-1, self.dim)
        return torch.from_numpy(ub.astype(self._numpy_dtype))
    
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
        if x is not None:
            
            u = self.fun_u(x)
            
            return u
        
        else:
            
            # x_mesh = np.linspace(self._lb, self._ub, self._N_test)
            # x = torch.from_numpy(x_mesh.reshape(-1,1).astype(self._numpy_dtype))
            
            x = self._gen_data.get_in(Nx_size=self._N_test, method='mesh')
            u = self.fun_u(x)
            
            return u, x
        
    ###########################################################################
    
    def fun_bd_err(self, model:torch.nn.Module, x_list:list[torch.tensor]) -> torch.tensor:
        '''
        Input:
            model: 
            x_list: list= [size(n,d)]*2d     
        Output:
            cond_bd: size(n*2d,1)
        '''
        
        x_bd = torch.cat(x_list, dim=0)
        u_true = self.get_test(x_bd)
        u_pred = model(x_bd)
        
        return u_true-u_pred
    
    ###########################################################################
        
    def fun_u(self, x:torch.tensor) -> torch.tensor:
        
        u = torch.zeros_like(x)
        
        sqrt2_torch = torch.tensor(math.sqrt(2), dtype = self._torch_dtype)
        
        cond1 = (x>=0) * (x< 1/(2*sqrt2_torch))
        cond2 = (x>= 1/(2*sqrt2_torch)) * (x< 0.5)
        cond3 = (x>=0.5) * (x<1- 1/(2*sqrt2_torch))
        cond4 = (x>= ( 1 - 1/(2*sqrt2_torch))) * (x<=1)
        
        x1 = x[cond1]
        x2 = x[cond2]
        x3 = x[cond3]
        x4 = x[cond4]
        
        u[cond1] = (100 - 50 * sqrt2_torch) *x1
        u[cond2] = 100 * x2 * (1 - x2) - 12.5
        u[cond3] = 100 * x3 * (1 - x3) - 12.5
        u[cond4] = (100 - 50 * sqrt2_torch) * (1 - x4)
        
        return u
        
    ###########################################################################
    
    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        Input:  
            x: size(?,d)
        Output: 
            f: size(?,1)
        '''
       
        f = torch.zeros_like(x)
        
        return f
    
    ###########################################################################
    
    def fun_psi(self, x:torch.tensor)->torch.tensor:
        '''
        Input:  
            x: size(?,d)
        Output: 
            psi: size(?,1)
        '''
        psi = torch.zeros_like(x)
        
        cond1 = (x>=0) * (x<0.25)
        cond2 = (x>= 0.25) * (x< 0.5)
        cond3 = (x>=0.5) * (x < 0.75)
        cond4 = (x>=0.75) * (x <= 1.0)
        
        x1 = x[cond1]
        x2 = x[cond2]
        x3 = x[cond3]
        x4 = x[cond4]
        
        psi[cond1] = 100 * x1**2
        psi[cond2] = 100 * x2 * (1 - x2) - 12.5
        psi[cond3] = 100 * x3 * (1 - x3) - 12.5
        psi[cond4] = 100 * (1-x4)**2
            
        return psi
    
    ###########################################################################

    def evi_pinn(self, model:torch.nn.Module, x:torch.tensor)->torch.tensor:
        '''
        The strong form residual
        Input: 
            model:
            x:size(?,d)
        Output: 
            The residual: size(?,1)
        '''
        #######################################################################
        
        x = Variable(x, requires_grad=True)
    
        u = model(x)
        du = self._grad_u(x, u)
        Lu = self._div_u(x, du)
        Au = -Lu
        
        #######################################################################
        
        eta = self._eta
        psi = self.fun_psi(x)
        f = self.fun_f(x)
        rhs = torch.nn.functional.relu(u-eta*Au+eta*f-psi)+psi
        
        #######################################################################
        eq = rhs-u
        
        return eq

    ###########################################################################
    

###############################################################################
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
            dtype: np.float
        '''
        
        #######################################################################
        
        self._name = args['problem_name']
        self._dim = args['problem_dim']
        self._domain = args['problem_domain']
        self._eta = 1e-1
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
    
    def fun_u(self, x:torch.tensor)->torch.tensor:
        
        beta = torch.tensor(0.02376)
        u = torch.zeros_like(x)
        
        cond1 = (x>= -1) * (x< -0.5-beta)
        cond2 = (x>= -0.5-beta) * (x< -0.5)
        cond3 = (x>= -0.5) * (x< 0.5)
        cond4 = (x>= 0.5) * (x<= 0.5+beta)
        cond5 = (x>= 0.5+beta) * (x<= 1)
        
        x1 = x[cond1]
        x2 = x[cond2]
        x3 = x[cond3]
        x4 = x[cond4]
        x5 = x[cond5]
        
        u[cond1] = self.fun_psi(-beta-0.5)*(x1+1)/(0.5-beta)
        u[cond2] = self.fun_psi(x2)
        u[cond3] = torch.ones_like(x3)
        u[cond4] = self.fun_psi(x4)
        u[cond5] = self.fun_psi(beta+0.5)*(x5-1)/(beta-0.5)
        
        return u
    
    def fun_mu(self, x:torch.tensor)->torch.tensor:
        
        mu = torch.zeros_like(x)
        cond = x>0
        mu[cond] = torch.exp(-1/x[cond]) 
        
        return mu
    
    def fun_phi(self, x:torch.tensor)->torch.tensor:
        
        phi = self.fun_mu(0.4-torch.abs(x))/(self.fun_mu(0.4-torch.abs(x))+self.fun_mu(torch.abs(x)-0.3))
        
        return phi
    
    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        Input:  
            x: size(?,d)
        Output: 
            f: size(?,1)
        '''
       
        f = torch.zeros_like(x)
        
        return f
    
    def fun_psi(self, x:torch.tensor)->torch.tensor:
        '''
        Input:  
            x: size(?,d)
        Output: 
            psi: size(?,1)
        '''
        alpha = 0.4
        psi = torch.zeros_like(x)
        
        cond1 = (x>=-1) * (x<0)
        cond2 = (x>= 0) * (x<=1)
        
        x1 = x[cond1]
        x2 = x[cond2]

        
        psi[cond1] = self.fun_phi(x1+0.5)*(1.5-12*torch.pow(torch.abs(x1+0.5), 2-alpha))-0.5
        psi[cond2] = self.fun_phi(x2-0.5)*(1.5-12*torch.pow(torch.abs(x2-0.5), 2-alpha))-0.5
            
        return psi
    

    
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

    

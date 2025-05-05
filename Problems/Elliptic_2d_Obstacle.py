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
        self._eta = 1e-1
        self._R = args['problem_R']
        self._N_test = args['problem_N_test']
        self._domain = args['problem_domain']

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
        
        rstar = torch.tensor(0.6979651482)
        
        rsquared = (x[:, 0:1])**2+ (x[:, 1:2])**2
        r = torch.sqrt(rsquared)
        u = torch.zeros_like(r)

        cond1 = (r<=rstar)
        cond2 = (r>rstar)

        r1 = r[cond1]
        r2 = r[cond2]

        u[cond1] = torch.sqrt(1-r1**2)
        u[cond2] = -rstar**2*torch.log(r2/2)/torch.sqrt(1-rstar**2)

        return u

    
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

        rsquared = (x[:, 0:1])**2+ (x[:, 1:])**2
        cond = (rsquared<=1)
        
        psi = -1*torch.ones_like(rsquared)
        psi[cond] = torch.sqrt(1-rsquared[cond])
            
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

    

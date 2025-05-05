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
        self._R = args['problem_R']
        self._tau = torch.tensor(args['problem_tau'])
        self._domain = args['problem_domain']
        self._one = torch.tensor(1.0)
        
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
    
    def fun_bd_err(self, model:torch.nn.Module, x_list:list[torch.tensor]) -> torch.tensor:
        '''
        Input:
            model: 
            x_list: list= [size(n,d)]*2d    
            [lb1, x2], [ub1, x2], [x1, lb2], [x2, ub2]
        Output:
            cond_bd: size(n*2d,1)
        '''
        #######################################################################
        '''
           \Gamma = {1}*[0,1]
        '''
        # tau = self._tau
        # x_bd2 = x_list[1]
        # x_bd134 = torch.cat([x_list[0], x_list[2], x_list[3]], dim=0)
        # # x_bd134 = torch.cat(x_list, dim=0)
        
        # upred_bd134 = model(x_bd134)[:,0:1]
        # utrue_bd134 = self.get_test(x_bd134)
        
        # x_bd2 = Variable(x_bd2, requires_grad=True)
        # up_bd2 = model(x_bd2)
        # u, p = up_bd2[:, 0:1], up_bd2[:, 1:]
        # dudx1 = self._grad_u(x_bd2, u)[:,0:1]
        
        
        # eq1 = dudx1+tau*p
        # eq2 = p*u-tau*torch.abs(u)
        # eq3 = torch.max(tau, torch.abs(p))-tau
        # # eq4 = upred_bd134-utrue_bd134
        

        # eq = torch.sqrt(torch.mean(eq1**2+eq2**2+eq3**2)+torch.mean(eq4**2))
        # eq = torch.sqrt(torch.mean(eq1**2+eq2**2+eq3**2))
        
        #######################################################################
        '''
           \Gamma = \Patial\Omega, \Omega = [0,1]*[0,1]
        '''
        tau = self._tau
        x_bd2 = x_list[1]
        
        x_bd2 = Variable(x_bd2, requires_grad=True)
        up_bd2 = model(x_bd2)
        u, p = up_bd2[:, 0:1], up_bd2[:, 1:]
        dudx1 = self._grad_u(x_bd2, u)[:,0:1]
        pnew = tau*p/torch.max(tau, torch.abs(p))
        
        eq1 = dudx1+tau*pnew
        eq2 = pnew*u-tau*torch.abs(u)
 
        eq = torch.sqrt(torch.mean(eq1**2+eq2**2))

        return eq
    
    def fun_u(self, x:torch.tensor)->torch.tensor:
        
        tau = self._tau
        one = self._one
        pi = torch.pi
        x1, x2 = x[:,0:1], x[:,1:2]
        
        u = tau*(torch.sin(x1)-x1*torch.sin(one))*torch.sin(2*pi*x2)
        
        # u = tau*torch.sin(pi*x1)*torch.sin(pi*x2)
       
        return u

    
    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        Input:  
            x: size(?,d)
        Output: 
            f: size(?,1)
        '''
        
        tau = self._tau
        one = self._one
        pi = torch.pi
        x1, x2 = x[:,0:1], x[:,1:2]
        
        f = tau*((2+4*pi**2)*torch.sin(x1)-(1+4*pi**2)*x1*torch.sin(one))*torch.sin(2*pi*x2)
        
        # f = tau*(2*pi**2+1)*torch.sin(pi*x1)*torch.sin(pi*x2)
        
        return f
    

    
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
        f = self.fun_f(x)
        
        u = model(x)[:, 0:1]
        
        du = self._grad_u(x, u)
        Lu = self._div_u(x, du)
        Au = -Lu+u
        
        eq = Au-f

        #######################################################################
        
        return eq

    

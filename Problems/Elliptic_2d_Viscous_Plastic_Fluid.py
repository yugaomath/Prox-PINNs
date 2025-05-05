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
        self._constant = args['problem_c']
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
            
            x_mesh = np.linspace(-self._R, self._R, self._N_test)
            x_mesh, y_mesh = np.meshgrid(x_mesh, x_mesh)
            xnp = np.concatenate([x_mesh.reshape(-1,1), y_mesh.reshape(-1,1)], axis=1)
            x = torch.from_numpy(xnp.astype(self._numpy_dtype))
            
            ind = torch.norm(x, dim=1)<=self._R
            x = x[ind]
            
            # data = self.get_point_disk(10000, 100)
            # x = data['x_in']
            
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
        #
        cond_uppred = model(x_bd)
        cond_pred = cond_uppred[:, 0:1]
        
        cond_true = self.get_test(x_bd)

        return cond_pred - cond_true
    
    def fun_u(self, x:torch.tensor)->torch.tensor:
        
        c = self._constant
        R = self._R
        tau = self._tau
        r = torch.norm(x, p=2, dim=1, keepdim=True)
        
        if c*R<=2*tau:
            
            u = torch.zeros_like(r)
            
        else:
            
            u = (R-r)/2*(c/2*(R+r)-2*tau)
            
            R1 = 2*tau/c
            cond = (r<=R1)
            u[cond] = (R-R1)/2*(c/2*(R+R1)-2*tau)

        return u

    
    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        Input:  
            x: size(?,d)
        Output: 
            f: size(?,1)
        '''
        # one_torch = torch.tensor(1, dtype = self._torch_type)
        
        x1 = x[:, 0:1]
        c = self._constant
        f = c*torch.ones_like(x1)
        
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
        tau = self._tau
        
        #######################################################################
        '''
           case I: p is constructed by neural network
        '''
        # u_p = model(x)
        # u, p = u_p[:,0:1], u_p[:,1:3]
        
        # du = self._grad_u(x, u)
        # divp = self._div_u(x, p)
        
        # Lu = self._div_u(x, du)
        # Au = -Lu
        
        # du_norm = torch.norm(du, p=2, dim=1, keepdim=True)
        # p_norm = torch.norm(p, p=2, dim=1, keepdim=True)
        
        
        # eq1 = p[:,0:1]*du[:,0:1]+p[:,1:2]*du[:,1:2]-tau*du_norm
        # indx = du_norm<torch.tensor(1e-15)
        # eq1[indx] = torch.max(tau, p_norm[indx])-tau
        
        # eq2 = Au-f-divp
        # eq = torch.sqrt(eq1**2+eq2**2)

        #######################################################################
        '''
           case II: p = tau*NN/max(tau, NN) with exact form
        '''
        
        u_p = model(x)
        u, p = u_p[:,0:1], u_p[:,1:3]
        
        du = self._grad_u(x, u)
        # divp = self._div_u(x, p)
        
        
        Lu = self._div_u(x, du)
        Au = -Lu
        
        du_norm = torch.norm(du, p=2, dim=1, keepdim=True)
        p_norm = torch.norm(p, p=2, dim=1, keepdim=True)
        
        
        pnew = tau*p/torch.max(tau, p_norm)
        
        eq1 = pnew[:,0:1]*du[:,0:1]+pnew[:,1:2]*du[:,1:2]-tau*du_norm
        
        divpnew = self._div_u(x, pnew)
        eq2 = Au-f-divpnew
        eq = torch.sqrt(eq1**2+eq2**2)
        
        return eq

    

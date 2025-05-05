
import numpy as np
import torch
import torch.nn as nn
'''
This file is inspired by the work of Yaohua Zhang in ParticleWAN.
'''
class FeedForward(nn.Module):
    '''
    '''
    def __init__(self, d_in:int=1, d_out:int=1, hidden_size:int=100, hidden_layers=3, 
                 activation:str='tanh', **kwargs):
        
        super(FeedForward, self).__init__()
        
        
        # Activation
        if activation=='relu':
            self.activation = torch.nn.ReLU()
        elif activation=='elu':
            self.activation = torch.nn.ELU()
        elif activation=='softplus':
            self.activation = torch.nn.Softplus()
        elif activation=='sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation=='tanh':
            self.activation = torch.nn.Tanh()
        else:
            raise NotImplementedError
            
        # Network Sequential
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        self.fc_hidden_list = nn.ModuleList()
        
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
            
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)
            
        

    def forward(self, x):
        #
        x = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        x = self.activation(self.fc_in(x))
        ############################
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(fc_hidden(x)) + x

        return self.fc_out(x)

class FeedForward_Partial_Constraint(nn.Module):
    '''
    '''
    def __init__(self, d_in:int=1, d_out:int=1, hidden_size:int=100, hidden_layers=3, 
                 activation:str='tanh', **kwargs):
        
        super(FeedForward_Constraint, self).__init__()
        
        self.din = d_in
        self.fun_u = kwargs['fun_u']
        self.R = kwargs['R']
        self.pde_domain = kwargs['pde_domain']
        
        # Activation
        if activation=='relu':
            self.activation = torch.nn.ReLU()
        elif activation=='elu':
            self.activation = torch.nn.ELU()
        elif activation=='softplus':
            self.activation = torch.nn.Softplus()
        elif activation=='sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation=='tanh':
            self.activation = torch.nn.Tanh()
        else:
            raise NotImplementedError
            
        # Network Sequential
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        self.fc_hidden_list = nn.ModuleList()
        
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)
        

    def gd(self, x):
        
        if self.din ==1:
            
            u_lb = self.fun_u(self.lb)
            u_ub = self.fun_u(self.ub)
            
            val = (self.ub - x) / (self.ub - self.lb) * u_ub + \
                  (x - self.lb) / (self.ub - self.lb) * u_lb
                  
            return val
        
        elif self.din ==2:
            
            x1, x2 = x[:, 0:1], x[:, 1:2]
            
            lb1, lb2 = self.lb[:,0:1], self.lb[:,1:2] # a,c 
            ub1, ub2 = self.ub[:,0:1], self.ub[:,1:2] # b,d
            
            
            lb1, lb2 = lb1.expand(x1.shape[0], -1), lb2.expand(x1.shape[0], -1)
            ub1, ub2 = ub1.expand(x1.shape[0], -1), ub2.expand(x1.shape[0], -1)
            

            x_lb = [torch.cat([lb1, x2], dim=1), torch.cat([x1, lb2], dim=1)] #  (a, y), (x,c)
            x_ub = [torch.cat([ub1, x2], dim=1), torch.cat([x1, ub2], dim=1)] #  (b, y), (x,d)
            
            u_lb = [self.fun_u(x_lb[0]), self.fun_u(x_lb[1])] # ua(y), uc(x)
            u_ub = [self.fun_u(x_ub[0]), self.fun_u(x_ub[1])] # ub(y), ud(x)
                      
            Interpolation = (ub1-x1)/(ub1-lb1)*u_lb[0]+\
                            (ub2-x2)/(ub2-lb2)*u_lb[1]+(x2-lb2)/(ub2-lb2)*u_ub[1]
            
    
            corner_ll = torch.cat([lb1, lb2], dim=1)  # (a, c)
            corner_lu = torch.cat([lb1, ub2], dim=1)  # (a, d)
            
        
            u_corner_ll = self.fun_u(corner_ll)
            u_corner_lu = self.fun_u(corner_lu)
            
            weight_ll = ((ub1 - x1) / (ub1 - lb1)) * ((ub2 - x2) / (ub2 - lb2))  
            weight_lu = ((ub1 - x1) / (ub1 - lb1)) * ((x2 - lb2) / (ub2 - lb2))  
            
            # 
            corner_terms = (weight_ll * u_corner_ll + weight_lu * u_corner_lu )
            
            val =  Interpolation-corner_terms
            
            return val
                
        else:
            
            raise NotImplementedError
            
    def dist(self, x):
        
        if self.din ==1:
            
            val = (x - self.lb)*(self.ub - x) 
                  
            return val
        
        elif self.din ==2:
 
            x1, x2 = x[:, 0:1], x[:, 1:2]
            lb1, lb2 = self.lb[:,0:1], self.lb[:,1:2]
            ub1, ub2 = self.ub[:,0:1], self.ub[:,1:2]
            val = (x1-lb1)*(x2-lb2)*(ub2-x2) 
            max_val = (ub1-lb1)*(ub2-lb2)**2/4
            val = val/max_val
            
            return val

        
        else:
            
            raise NotImplementedError

    def forward(self, x):
        
        #######################################################################
        
        z = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        z = self.activation(self.fc_in(z))
        
        #######################################################################
        
        for fc_hidden in self.fc_hidden_list:
            
            z = self.activation(fc_hidden(z)) + z
        
        val = self.fc_out(z)
        val_first_col = self.gd(x) + self.dist(x) * val[:, 0:1]
        val = torch.cat([val_first_col, val[:, 1:]], dim=1)  
        
        return val
    
        #######################################################################
        
class FeedForward_Constraint(nn.Module):
    '''
    '''
    def __init__(self, d_in:int=1, d_out:int=1, hidden_size:int=100, hidden_layers=3, 
                 activation:str='tanh', **kwargs):
        
        super(FeedForward_Constraint, self).__init__()
        
        self.din = d_in
        self.fun_u = kwargs['fun_u']
        self.R = kwargs['R']
        self.pde_domain = kwargs['pde_domain']
        
        # Activation
        if activation=='relu':
            self.activation = torch.nn.ReLU()
        elif activation=='elu':
            self.activation = torch.nn.ELU()
        elif activation=='softplus':
            self.activation = torch.nn.Softplus()
        elif activation=='sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation=='tanh':
            self.activation = torch.nn.Tanh()
        else:
            raise NotImplementedError
            
        # Network Sequential
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        self.fc_hidden_list = nn.ModuleList()
        
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)
        

    def gd(self, x):
        
        if self.din ==1:
            
            u_lb = self.fun_u(self.lb)
            u_ub = self.fun_u(self.ub)
            
            val = (self.ub - x) / (self.ub - self.lb) * u_ub + \
                  (x - self.lb) / (self.ub - self.lb) * u_lb
                  
            return val
        
        elif self.din ==2:
            
            if self.pde_domain == 'circle':
                
                xR = torch.tensor(np.array([[self.R, 0]]))
             
                return self.fun_u(xR)
        
            elif self.pde_domain == 'rect':

                x1, x2 = x[:, 0:1], x[:, 1:2]
                
                lb1, lb2 = self.lb[:,0:1], self.lb[:,1:2] # a,c 
                ub1, ub2 = self.ub[:,0:1], self.ub[:,1:2] # b,d
                
                
                lb1, lb2 = lb1.expand(x1.shape[0], -1), lb2.expand(x1.shape[0], -1)
                ub1, ub2 = ub1.expand(x1.shape[0], -1), ub2.expand(x1.shape[0], -1)
                
    
                x_lb = [torch.cat([lb1, x2], dim=1), torch.cat([x1, lb2], dim=1)] #  (a, y), (x,c)
                x_ub = [torch.cat([ub1, x2], dim=1), torch.cat([x1, ub2], dim=1)] #  (b, y), (x,d)
                
                u_lb = [self.fun_u(x_lb[0]), self.fun_u(x_lb[1])] # ua(y), uc(x)
                u_ub = [self.fun_u(x_ub[0]), self.fun_u(x_ub[1])] # ub(y), ud(x)
                          
                Interpolation = (ub1-x1)/(ub1-lb1)*u_lb[0]+(x1-lb1)/(ub1-lb1)*u_ub[0]+\
                                (ub2-x2)/(ub2-lb2)*u_lb[1]+(x2-lb2)/(ub2-lb2)*u_ub[1]
                
        
                corner_ll = torch.cat([lb1, lb2], dim=1)  # (a, c)
                corner_rl = torch.cat([ub1, lb2], dim=1)  # (b, c)
                corner_lu = torch.cat([lb1, ub2], dim=1)  # (a, d)
                corner_ru = torch.cat([ub1, ub2], dim=1)  # (b, d)
                
            
                u_corner_ll = self.fun_u(corner_ll)
                u_corner_rl = self.fun_u(corner_rl)
                u_corner_lu = self.fun_u(corner_lu)
                u_corner_ru = self.fun_u(corner_ru)
                
                weight_ll = ((ub1 - x1) / (ub1 - lb1)) * ((ub2 - x2) / (ub2 - lb2))  
                weight_rl = ((x1 - lb1) / (ub1 - lb1)) * ((ub2 - x2) / (ub2 - lb2))  
                weight_lu = ((ub1 - x1) / (ub1 - lb1)) * ((x2 - lb2) / (ub2 - lb2))  
                weight_ru = ((x1 - lb1) / (ub1 - lb1)) * ((x2 - lb2) / (ub2 - lb2))  
                
                # 
                corner_terms = (weight_ll * u_corner_ll +weight_rl * u_corner_rl +weight_lu * u_corner_lu +weight_ru * u_corner_ru)
                
                val =  Interpolation-corner_terms
                
                return val
            else:
                raise NotImplementedError

        else:
            
            raise NotImplementedError
            
    def dist(self, x):
        
        if self.din ==1:
            
            val = (x - self.lb)*(self.ub - x) 
            max_val = (self.ub - self.lb)**2/4
            val = val/max_val
                  
            return val
        
        elif self.din ==2:
            
            if self.pde_domain == 'circle':
                
                rsquare = x[:,0:1]**2+x[:,1:2]**2
                val = ((self.R)**2-rsquare)/(self.R)**2
                
                return val

            elif self.pde_domain == 'rect':
            
                x1, x2 = x[:, 0:1], x[:, 1:2]
                lb1, lb2 = self.lb[:,0:1], self.lb[:,1:2]
                ub1, ub2 = self.ub[:,0:1], self.ub[:,1:2]
                val = (x1-lb1)*(ub1-x1)*(x2-lb2)*(ub2-x2)
                
                max_val = (ub1-lb1)**2*(ub2-lb2)**2/16
                val = val/max_val
            
                return val
            else:
                
                raise NotImplementedError
        
        else:
            
            raise NotImplementedError

    def forward(self, x):
        
        #######################################################################
        
        z = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        z = self.activation(self.fc_in(z))
        
        #######################################################################
        
        for fc_hidden in self.fc_hidden_list:
            
            z = self.activation(fc_hidden(z)) + z
        
        val = self.fc_out(z)
        val_first_col = self.gd(x) + self.dist(x) * val[:, 0:1]
        val = torch.cat([val_first_col, val[:, 1:]], dim=1)  
        
        return val
    
        #######################################################################
    
class Model():
    '''
    '''
    def __init__(self, model_type:str, device=None, dtype:torch.dtype = torch.float64):
        
        self.model_type = model_type
        self.device = device
        torch.set_default_dtype(dtype)
    
    def get_model(self, d_in:int=1, d_out:int=1, h_size:int=100, l_size:int=3, activation:str='tanh', **kwargs):
        
        if self.model_type=='FeedForward':
            return FeedForward(d_in=d_in, d_out=d_out, hidden_size=h_size, hidden_layers=l_size, 
                               activation=activation, **kwargs).to(self.device)
        
        
        elif self.model_type=='FeedForward_Constraint':
            return FeedForward_Constraint(d_in=d_in, d_out=d_out, hidden_size=h_size, hidden_layers=l_size,
                                          activation=activation, **kwargs).to(self.device)

        elif self.model_type=='FeedForward_Partial_Constraint':
            return FeedForward_Constraint(d_in=d_in, d_out=d_out, hidden_size=h_size, hidden_layers=l_size,
                                          activation=activation, **kwargs).to(self.device)
        
        else:
            raise NotImplementedError
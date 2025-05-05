
import torch
import sys

class Error():

    def __init__(self):
        pass 

    def L2_and_Linf_error(self, model:torch.nn.Module, x:torch.tensor, u:torch.tensor)->torch.tensor:
        '''
        Input: 
            model: network model
            x: size(?,d)
            u: size(?,k)
        Output: 
            err:    size(1,k)
            u_pred: size(?,k)
        '''
        with torch.no_grad():
            
            up_pred = model(x)
            u_pred = up_pred[:,0:1]
            u_abs = torch.abs(u)
            error = torch.abs(u_pred-u)
            
            # l2_error = torch.mean( error**2, dim=0) \
            #     / (torch.mean(u**2, dim=0) + sys.float_info.epsilon)
            
            l2_error = torch.norm(error, p=2)\
                /(torch.norm(u, p=2)+ sys.float_info.epsilon)
            
            linf_error = torch.max(error) \
                / (torch.max(u_abs) + sys.float_info.epsilon)
                
                
        return l2_error, linf_error, u_pred



###############################################################################
import numpy as np
import torch
from datetime import datetime
from Utils.Set_Seed import set_numpy_and_torch_seed
from Utils.Example import Example
from Solvers.DL4EVI_1d_Obstacle import Solver
from Problems.Elliptic_1d_Piecewise_Smooth import Problem
###############################################################################
   

if __name__=='__main__':
    
    ###########################################################################

    seed = 2025
    set_numpy_and_torch_seed(seed)
        
    numpy_dtype = np.float64
    torch_dtype = torch.float64        
    
    ###########################################################################
    
    problem_name = 'Elliptic_1d_Piecewise_Smooth'
    
    args_problem = {'problem_name': problem_name,
                    'problem_dim': 1,
                    'problem_lb': [-1.],
                    'problem_ub': [1.],
                    'problem_domain': 'line_segment',
                    'problem_N_test': 10000,
                    'numpy_dtype': numpy_dtype,
                    'torch_dtype': torch_dtype}
    
    demo_problem = Problem(**args_problem)
    
    ###########################################################################
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n  Using Device: {}'.format(device))
    try:
        print(f'{torch.cuda.get_device_name(0)}')
    except:
        pass
    
    args_solver = {'device': device,
                   'seed': 2025,
                   'int_method': 'mesh',
                   'N_int': 50,
                   'N_bd_each_face': 1,
                   'topK': 50,
                   'NN_in': args_problem['problem_dim'],
                   'NN_out': args_problem['problem_dim'],
                   'lr_adam': 1e-3, 
                   'lr_lbfgs': 1e-2, 
                   'lr_Decay': 2.,
                   'Iter_adam': 10000,
                   'Iter_lbfgs': 0,
                   'loss_weight': {'eq':1., 'bd':5.},
                   'activation': 'tanh',
                   'hidden_layer': 3,
                   'hidden_width': 100,     
                   'model_type': 'FeedForward_Constraint', # 'FeedForward_Constraint' or 'FeedForward'
                   'eta': 1e-3,
                   'numpy_dtype': numpy_dtype,
                   'torch_dtype': torch_dtype}
    
    demo_solver = Solver(problem = demo_problem, **args_solver)
    
    ###########################################################################

    curren_time = datetime.now()
    time_str = curren_time.strftime("%Y_%m_%d_%H")
    
    # time_str = '2025_03_12_19'
        
    solver_names = 'DL4EVI'
    path = f"./SavedData/{problem_name}/"
    
    demo_example = Example(idx = time_str, path = path, solver_name = solver_names)
    
    ###########################################################################
    
    print(f'  Solve Problem: {problem_name} ')
    
    demo_example.run(solver = demo_solver,  load_type = 'model_best_loss', status = 'train')
    demo_example.run(solver = demo_solver,  load_type = 'model_best_loss', status = 'pred')
    demo_example.run(solver = demo_solver,  load_type = 'model_best_loss', status = 'plot')
    
    ###########################################################################
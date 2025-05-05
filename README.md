# Prox-PINNs for Solving Elliptic Variational Inequalities

## Requirements
- Python 3.x
- NumPy (`numpy`)
- SciPy (`scipy`)
- PyTorch (`torch`)
- Matplotlib (`matplotlib`)

## Quick Start Guide

###  1. Problem Definition
Create a problem file in `Problems/` directory with these components:

```python
def __init__(self, np_type=np.float64, torch_type=torch.float64, **args):
    self.domain = [...]  # Define problem domain

def fun_u(self, x: torch.Tensor) -> torch.Tensor:
    return ...  # Exact solution implementation

def fun_f(self, x: torch.Tensor) -> torch.Tensor:
    return ...  # Source term implementation

def evi_pinn(self, model:torch.nn.Module, x:torch.tensor)->torch.tensor:
```


### 2. Solver Implementation
In the 'Solvers/' directory, you will find several examples of solvers. Within each solver, you can examine the function

- ```
  _load(self, model:torch.nn.Module, load_path:str, load_type:str)->None:
  ```

- ```
  _save(self, model:torch.nn.Module, save_path:str, save_type:str)->None:
  ```

- ```
  get_net(self)->None:
  ```

- ```
  get_loss(self, data_point:dict, **args):
  ```

- ```
  train(self, save_path:str, load_type:str=None)->None:
  ```

- ```
  predict(self, load_path:str, load_type:str)->None:
  ```

- ```
  plot_fig(self, load_path:str)->None:
  ```

### 3. Solve the Problem

In the main directory, you will find several examples. For each example, you need to set

- choose the problem, e.g. ,

  ```
  from Solvers.DL4EVI_1d_Obstacle import Solver
  ```

- choose the solver, e.g. ,

  ```
  from Problems.Elliptic_1d_Symmetric import Problem
  ```

- set the parameters for problem

  ```
  args_problem =  { 'problem_name': problem_name,
                    'problem_dim': 1,
                    'problem_lb': [0.],
                    'problem_ub': [1.],
                    'problem_domain': 'line_segment',
                    'problem_N_test': 1000,
                    'numpy_dtype': numpy_dtype,
                    'torch_dtype': torch_dtype}
  ```


- set the parameters for solver

  ```
  args_solver =  { 'device': device,
                   'seed': 2025,
                   'int_method': 'mesh',
                   'N_int': 50,
                   'N_bd_each_face': 1,
                   'topK': 50,               # when topK=N_int, the topK strategy was not used
                   'NN_in': args_problem['problem_dim'],
                   'NN_out': args_problem['problem_dim'],
                   'lr_adam': 1e-3, 
                   'lr_lbfgs': 1e-1, 
                   'lr_Decay': 2.,
                   'Iter_adam': 10000,
                   'Iter_lbfgs': 0,             # L-BFGS was not employed for refinement 
                   'loss_weight': {'eq':1., 'bd':5.}, # bd was not used for hard constraint
                   'activation': 'tanh',
                   'hidden_layer': 3,
                   'hidden_width': 100,
                   'model_type': 'FeedForward_Constraint', # 'FeedForward', 'FeedForward_Constraint', 'FeedForward_Constraint_Partial_Constraint' 
                   'eta': 1e-3,
                   'numpy_dtype': numpy_dtype,
                   'torch_dtype': torch_dtype}
  ```

- If you have a saved model, you only need to run

  ```
  demo_example.run(solver = demo_solver,  load_type = 'model_best_loss', status = 'pred')
  demo_example.run(solver = demo_solver,  load_type = 'model_best_loss', status = 'plot')
  ```

  If you have a saved model and the corresponding data, you only need to run

  ```
  demo_example.run(solver = demo_solver,  load_type = 'model_best_loss', status = 'plot')
  ```

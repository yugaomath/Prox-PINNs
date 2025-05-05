# Prox-PINNs for Solving Elliptic Variational Inequalities

## Requirements

## Requirements

The following dependencies are required to run this project:

- Python 3.x
- NumPy (`numpy`)
- SciPy (`scipy`)
- PyTorch (`torch`)
- Matplotlib (`matplotlib`)

## Quick Start Guide

### 1. Problem Definition
Create a problem file in `Problems/` directory with these essential components:

```python
def __init__(self, np_type=np.float64, torch_type=torch.float64, **args):
    # Define problem domain and parameters
    self.domain = [...]  

def fun_u(self, x: torch.Tensor) -> torch.Tensor:
    # Implement exact solution
    return ...  

def fun_f(self, x: torch.Tensor) -> torch.Tensor:
    # Implement source term
    return ...
#### 2. Choose the Solver

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

Note that:

- obstacle problem

  ```
  DL4EVI_1d.py
  DL4EVI_2d.py
  ```

- introduced multiplier

  ```python
  DL4EVI_1d_Multiplier.py
  DL4EVI_2d_Multiplier.py
  ```

#### 3. Solve the Problem

In the main directory, you will find several examples. For each example, you need to set

- choose the problem, e.g. ,

  ```
  from Solvers.DL4EVI_1d import Solver
  ```

- choose the solver, e.g. ,

  ```
  from Problems.Elliptic_1d_Non_Symmetric import Problem
  ```

- set the parameters in dictionary args 

      args = {'device': device,
              'int_method': 'mesh',
              'N_int': 200,
              'N_bd_each_face': 1,
              'topK': 200,
              'lr': 1e-3,   
              'lr_Decay': 2.,
              'maxIter': 10000,
              'loss_weight': {'eq':1., 'bd':5.},
              'activation': 'tanh',
              'hidden_width': 50,
              'hidden_layer': 4,
              'model_type': 'FeedForward',
              'data_type': {'numpy': np.float64, 'torch': torch.float64},
              }

- If you have a saved model, you only need to run

  ```
  demo_example.run(solver = demo_solver,  load_type = 'model_best_loss', status = 'pred')
  demo_example.run(solver = demo_solver,  load_type = 'model_best_loss', status = 'plot')
  ```

  If you have a saved model and the corresponding data, you only need to run

  ```
  demo_example.run(solver = demo_solver,  load_type = 'model_best_loss', status = 'plot')
  ```

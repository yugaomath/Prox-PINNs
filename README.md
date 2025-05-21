# Prox-PINNs for Solving Elliptic Variational Inequalities

This repository provides an implementation of **Prox-PINNs**, a novel deep learning algorithmic framework that integrates proximal operators with physics-informed neural networks (**PINNs**) to solve a broad class of elliptic variational inequalities (**EVIs**).

📄 **Paper:** ["Prox-PINNs: A Deep Learning Algorithmic Framework for Elliptic Variational Inequalities"](https://arxiv.org/abs/2505.14430) 

👥 **Authors:** Yu Gao, Yongcun Song, Zhiyu Tan, Hangrui Yue, and Shangzhi Zeng

## Requirements
- Python 3.x
- NumPy
- SciPy
- PyTorch 
- Matplotlib

## Quick Start Guide

### 1. Problem Definition
Create a problem file in the `Problems/` directory with the following components:

```python
def __init__(self, np_type=np.float64, torch_type=torch.float64, **args):
    self.domain = [...]  # Define problem domain [lower_bound, upper_bound]

def fun_u(self, x: torch.Tensor) -> torch.Tensor:
    return ...  # Implement exact solution

def fun_f(self, x: torch.Tensor) -> torch.Tensor:
    return ...  # Implement source term

def evi_pinn(self, model: torch.nn.Module, x: torch.tensor) -> torch.tensor:
    # Implement EVI-PINN specific calculations
```


### 2. Solver Implementation
In the `Solvers/` directory, implement the following key functions:

```python
def _load(self, model: torch.nn.Module, load_path: str, load_type: str) -> None:
        # Load model from specified path
    
def _save(self, model: torch.nn.Module, save_path: str, save_type: str) -> None:
        # Save model to specified path
    
def get_net(self) -> None:
        # Initialize neural network architecture
    
def get_loss(self, data_point: dict, **args):
        # Calculate loss function
    
def train(self, save_path: str, load_type: str = None) -> None:
        # Training procedure
    
def predict(self, load_path: str, load_type: str) -> None:
        # Make predictions using trained model
    
def plot_fig(self, load_path: str) -> None:
        # Generate visualization plots
```


### 3. Solve the Problem

In the main directory, several example scripts are available to guide you through solving a problem. Follow these steps:

- Import the problem and solver, e.g. ,

  ```python
  from Solvers.DL4EVI_1d_Obstacle import Solver
  from Problems.Elliptic_1d_Symmetric import Problem
  ```

- Set the parameters for problem, e.g. ,

  ```python
  args_problem =  { 'problem_name': problem_name,
                    'problem_dim': 1,
                    'problem_lb': [0.],
                    'problem_ub': [1.],
                    'problem_domain': 'line_segment',
                    'problem_N_test': 1000,
                    'numpy_dtype': numpy_dtype,
                    'torch_dtype': torch_dtype}
  ```


- Set the parameters for solver, e.g. ,

  ```python
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
- For new training:
  
  ```python
  demo_example.run(solver=demo_solver, load_type = 'model_best_loss',  status='train')
  ```

- For prediction with saved model:
  
  ```python
  demo_example.run(solver = demo_solver, load_type = 'model_best_loss', status = 'pred')
  ```

- For visualization with saved model:

  ```python
  demo_example.run(solver = demo_solver, load_type = 'model_best_loss', status = 'plot')
  ```
### 4. Results
All results, including trained models, prediction data, and visualizations, are stored in the `SavedData/`directory.

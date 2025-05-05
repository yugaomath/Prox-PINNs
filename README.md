# An Implementation of Prox-PINNs for Solving Elliptic Variational Inequalities

## Requirements

The following dependencies are required to run this project:

- Python 3.x
- NumPy (`numpy`)
- SciPy (`scipy`)
- PyTorch (`torch`)
- Matplotlib (`matplotlib`)

## How to Use

### 1. Define the Problem

Navigate to the `Problems/` directory where you'll find example problem definitions. Each problem should implement:

```python
def __init__(self, np_type=np.float64, torch_type=torch.float64, **args):
    """Initialize problem with data types and parameters"""
    
def fun_u(self, x: torch.Tensor) -> torch.Tensor:
    """Define exact solution"""
    
def fun_f(self, x: torch.Tensor) -> torch.Tensor:
    """Define source function"""
    
def evi_pinn(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Define loss function for the equation"""

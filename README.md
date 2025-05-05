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

### 2. Choose the Solver
The Solvers/ directory contains various solver implementations. Key components include:
_load(self, model: torch.nn.Module, load_path: str, load_type: str) -> None:
    """Load model from file"""

_save(self, model: torch.nn.Module, save_path: str, save_type: str) -> None:
    """Save model to file"""

get_net(self) -> None:
    """Initialize neural network architecture"""

get_loss(self, data_point: dict, **args):
    """Compute loss function"""

train(self, save_path: str, load_type: str = None) -> None:
    """Train the model"""

predict(self, load_path: str, load_type: str) -> None:
    """Make predictions with trained model"""

plot_fig(self, load_path: str) -> None:
    """Generate visualization plots"""

"""
Neural network models for damped harmonic oscillator.

Physics: x'' + 2ζω_n x' + ω_n² x = 0
"""
import torch
import torch.nn as nn


class BaselineNN(nn.Module):
    """Standard feedforward NN. No physics - learns purely from data."""
     
    def __init__(self, hidden_dims=[64, 64, 64]):
        super().__init__()
        layers = []
        in_dim = 1
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.Tanh()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, t):
        return self.net(t)


class PINN(nn.Module):
    """Physics-Informed NN. Same architecture, but uses physics loss."""
    
    def __init__(self, hidden_dims=[64, 64, 64]):
        super().__init__()
        layers = []
        in_dim = 1
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.Tanh()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, t):
        return self.net(t)
    
    def derivatives(self, t):
        """Compute x, dx/dt, d²x/dt² via autograd."""
        t = t.requires_grad_(True)
        x = self.forward(t)
        x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
        # If x_t is constant w.r.t. t (e.g., x=t), it won't require grad and the
        # second derivative is exactly zero.
        if x_t.requires_grad:
            x_tt = torch.autograd.grad(
                x_t, t, torch.ones_like(x_t), create_graph=True, allow_unused=True)[0]
            if x_tt is None:
                x_tt = torch.zeros_like(x_t)
        else:
            x_tt = torch.zeros_like(x_t)
        return x, x_t, x_tt


class BaselineLoss:
    """Standard MSE loss."""
    def __init__(self):
        self.mse = nn.MSELoss()
    
    def __call__(self, model, t_data, x_data, **kwargs):
        loss = self.mse(model(t_data), x_data)
        return loss, {'total': loss.item(), 'data': loss.item()}


class PINNLoss:
    """
    PINN loss: L = λ_data·L_data + λ_physics·L_ODE + λ_ic·L_IC
    
    L_ODE enforces: x'' + 2ζω_n x' + ω_n² x = 0
    """
    def __init__(self, zeta, omega_n, lambda_data=1.0, lambda_physics=1.0, lambda_ic=1.0):
        self.zeta = zeta
        self.omega_n = omega_n
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_ic = lambda_ic
        self.mse = nn.MSELoss()
    
    def __call__(self, model, t_data, x_data, t_physics, t_ic=None, x_ic=None, **kwargs):
        # Data loss
        loss_data = self.mse(model(t_data), x_data)
        
        # Physics loss: ODE residual should be zero
        x, x_t, x_tt = model.derivatives(t_physics)
        residual = x_tt + 2*self.zeta*self.omega_n*x_t + self.omega_n**2*x
        loss_physics = self.mse(residual, torch.zeros_like(residual))
        
        # Initial condition loss
        loss_ic = torch.tensor(0.0, device=t_data.device)
        if t_ic is not None and x_ic is not None:
            loss_ic = self.mse(model(t_ic), x_ic)
        
        total = self.lambda_data*loss_data + self.lambda_physics*loss_physics + self.lambda_ic*loss_ic
        
        return total, {
            'total': total.item(),
            'data': loss_data.item(),
            'physics': loss_physics.item(),
            'ic': loss_ic.item()}

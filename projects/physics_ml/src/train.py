"""
Training and evaluation for PINN vs Baseline comparison.
"""
import torch
import numpy as np
from torch.optim import Adam
from tqdm import tqdm

from .models import BaselineNN, PINN, BaselineLoss, PINNLoss


def train_model(model, loss_fn, data, n_epochs=5000, lr=1e-3, verbose=True):
    """Train a model and return loss history."""
    optimizer = Adam(model.parameters(), lr=lr)
    history = {'loss': []}
    
    iterator = tqdm(range(n_epochs), disable=not verbose, desc="Training")
    for epoch in iterator:
        optimizer.zero_grad() # reset gradients each epoch
        loss, loss_dict = loss_fn(
            model, data['t_train'], data['x_train'],
            t_physics=data.get('t_collocation'),
            t_ic=data.get('t_ic'), x_ic=data.get('x_ic'))
        loss.backward() # compute gradients
        optimizer.step() # update weights
        
        if verbose and epoch % 500 == 0:
            iterator.set_postfix(loss=f"{loss_dict['total']:.2e}")
        history['loss'].append(loss_dict['total'])
    
    return history


def evaluate_model(model, data):
    """Evaluate model and return metrics."""
    model.eval() # set model to evaluation mode (not relevant))
    with torch.no_grad(): #disable gradient computation
        pred = model(data['t_test'])
        true = data['x_test']
        mse = torch.mean((pred - true)**2).item()
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': torch.mean(torch.abs(pred - true)).item(),
        'predictions': model(data['t_full']).detach().cpu().numpy().flatten()} 


def train_and_evaluate(model_type, data, zeta, omega_n, n_epochs=5000, lr=1e-3,
                       lambda_physics=1.0, hidden_dims=[64, 64, 64], verbose=True):
    """Complete pipeline: create model, train, evaluate."""
    device = data['t_train'].device
    
    if model_type == 'baseline':
        model = BaselineNN(hidden_dims).to(device)
        loss_fn = BaselineLoss()
    else:  # pinn
        model = PINN(hidden_dims).to(device)
        loss_fn = PINNLoss(zeta, omega_n, lambda_physics=lambda_physics)
     
    history = train_model(model, loss_fn, data, n_epochs, lr, verbose)
    metrics = evaluate_model(model, data)
    
    return model, history, metrics

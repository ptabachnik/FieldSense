import numpy as np
import pandas as pd
import torch
from pathlib import Path


def download_kaggle_data():
    import kagglehub

    path = kagglehub.dataset_download("cici118/damped-harmonic-oscillator")
    return Path(path)
 

def load_kaggle_data(scenario='extrapolation'):
    """
    Load Kaggle damped oscillator dataset.
    
    Scenarios:
        - 'completion': fill missing data
        - 'reconstruction': reconstruct from partial
        - 'extrapolation': predict beyond training range
    
    Returns: t_train, x_train, t_test, x_test
    """
    path = download_kaggle_data()
    
    train_df = pd.read_csv(path / scenario / 'train.csv')
    test_df = pd.read_csv(path / scenario / 'test.csv')
    
    return (
        train_df['time'].values,
        train_df['displacement'].values,
        test_df['time'].values,
        test_df['displacement'].values)


def estimate_physics_params(t, x):
    """
    Estimate zeta and omega_n from damped oscillator data.
    
    Uses log-decrement method for damping ratio.
    """
    # Find peaks
    peaks_idx = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1] and x[i] > 0:
            peaks_idx.append(i)
    
    if len(peaks_idx) < 2:
        # Default values if can't estimate
        return 0.05, 0.1
    
    # Log decrement: δ = ln(x1/x2)
    x1, x2 = x[peaks_idx[0]], x[peaks_idx[1]]
    t1, t2 = t[peaks_idx[0]], t[peaks_idx[1]]
    
    if x2 <= 0 or x1 <= 0:
        return 0.05, 0.1
        
    delta = np.log(x1 / x2)
    
    # Damping ratio: ζ = δ / sqrt(4π² + δ²)
    zeta = delta / np.sqrt(4 * np.pi**2 + delta**2)
    
    # Damped period
    T_d = t2 - t1
    omega_d = 2 * np.pi / T_d
    
    # Natural frequency: ω_n = ω_d / sqrt(1 - ζ²)
    omega_n = omega_d / np.sqrt(1 - zeta**2) if zeta < 1 else omega_d
    
    return zeta, omega_n


def prepare_kaggle_datasets(scenario='extrapolation', n_collocation=200, device='cpu'):
    """Load Kaggle data and prepare for training."""
    t_train, x_train, t_test, x_test = load_kaggle_data(scenario)
    
    # Estimate physics parameters from training data
    zeta, omega_n = estimate_physics_params(t_train, x_train)
    
    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32, device=device).reshape(-1, 1)
    
    # Collocation points span both train and test range
    t_min = min(t_train.min(), t_test.min())
    t_max = max(t_train.max(), t_test.max())
    
    return {
        't_train': to_tensor(t_train),
        'x_train': to_tensor(x_train),
        't_test': to_tensor(t_test),
        'x_test': to_tensor(x_test),
        't_collocation': to_tensor(np.linspace(t_min, t_max, n_collocation)),
        't_ic': torch.tensor([[t_train.min()]],  dtype=torch.float32, device=device),
        'x_ic': torch.tensor([[x_train[0]]], dtype=torch.float32, device=device),
        't_full': to_tensor(np.concatenate([t_train, t_test])),
        'x_full': to_tensor(np.concatenate([x_train, x_test])),
        'zeta': zeta,
        'omega_n': omega_n}

 
def subsample_training_data(data, fraction, seed=42):
    """Subsample training data to test data efficiency."""
    np.random.seed(seed)
    n = len(data['t_train'])
    n_keep = max(1, int(n * fraction))
    idx = np.random.choice(n, n_keep, replace=False)
    
    new_data = data.copy()
    new_data['t_train'] = data['t_train'][idx]
    new_data['x_train'] = data['x_train'][idx]
    return new_data


def add_gaussian_noise(data, noise_std, seed=42):
    """
    Add white Gaussian noise to training data.
    
    Parameters
    ----------
    data : dict
        Dataset dict with 't_train', 'x_train', etc.
    noise_std : float
        Standard deviation of Gaussian noise (relative to data std).
        E.g., noise_std=0.1 means noise σ = 0.1 × data_std
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    new_data : dict
        Copy of data with noise added to x_train only.
        Test data remains clean (we evaluate on true values).
    """
    if noise_std <= 0:
        return data
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    new_data = data.copy()
    x_train = data['x_train']
    
    # Scale noise relative to data standard deviation
    data_std = x_train.std().item()
    absolute_noise_std = noise_std * data_std
    
    # Add Gaussian noise
    noise = torch.randn_like(x_train) * absolute_noise_std
    new_data['x_train'] = x_train + noise
    
    # Store noise info for reference
    new_data['noise_std'] = noise_std
    new_data['noise_std_absolute'] = absolute_noise_std
    
    return new_data

#!/usr/bin/env python3
"""
PINN vs Baseline NN - Kaggle Damped Harmonic Oscillator Dataset

Usage:
    python3 -m src.main                    # Full run, no noise
    python3 -m src.main --noise 0.2        # Full run, 20% noise
    python3 -m src.main --quick            # Quick test (10x fewer epochs)
    python3 -m src.main --noise 0.2 --quick
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from .data import prepare_kaggle_datasets, subsample_training_data, add_gaussian_noise
from .train import train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="PINN vs Baseline NN comparison")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Gaussian noise level (0.0=none, 0.1=10%%, 0.2=20%%, etc.)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode: 10x fewer epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


torch.manual_seed(42)
np.random.seed(42)

import matplotlib

matplotlib.use("Agg")  # headless-safe: we only save figures
import matplotlib.pyplot as plt


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1)


def save_run_plot(
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    data_full: dict,
    data_train: dict,
    model_baseline: torch.nn.Module,
    model_pinn: torch.nn.Module,
    rmse_baseline: float,
    rmse_pinn: float,
    zeta: float,
    omega_n: float,
    lambda_physics: float,
):
    """Save a single plot comparing Baseline vs PINN predictions."""
    device = data_full["t_train"].device
    t_min = min(data_full["t_train"].min().item(), data_full["t_test"].min().item())
    t_max = max(data_full["t_train"].max().item(), data_full["t_test"].max().item())
    t_grid = torch.linspace(t_min, t_max, 2000, device=device).reshape(-1, 1)

    model_baseline.eval()
    model_pinn.eval()
    with torch.no_grad():
        y_b = _to_np(model_baseline(t_grid))
        y_p = _to_np(model_pinn(t_grid))

    # Points
    t_train = _to_np(data_train["t_train"])
    x_train = _to_np(data_train["x_train"])
    t_test = _to_np(data_full["t_test"])
    x_test = _to_np(data_full["x_test"])
    order = np.argsort(t_test)
    t_test, x_test = t_test[order], x_test[order]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(_to_np(t_grid), y_b, label=f"Baseline (RMSE={rmse_baseline:.3f})", lw=2.0, color="#d62728")
    ax.plot(_to_np(t_grid), y_p, label=f"PINN (RMSE={rmse_pinn:.3f})", lw=2.0, color="#1f77b4")
    ax.scatter(t_test, x_test, s=10, alpha=0.35, color="black", label="True (test points)")
    ax.scatter(t_train, x_train, s=26, alpha=0.95, color="#2ca02c", label="Train points")

    ax.set_title(
        f"{title}\n"
        f"{subtitle}\n"
        f"Physics: ζ={zeta:.4f}, ωₙ={omega_n:.4f} | λ_physics={lambda_physics}"
    )
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("displacement (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    
    # Apply seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Epoch multiplier for quick mode
    epoch_mult = 0.1 if args.quick else 1.0
    
    # --- Header ---
    print("\n" + "=" * 70)
    print("  PINN vs Baseline NN — Kaggle Damped Oscillator")
    print("  ODE: x'' + 2ζωₙx' + ωₙ²x = 0")
    print("=" * 70)

    # --- Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = Path("outputs") / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    noise_str = f"noise={int(args.noise*100)}%" if args.noise > 0 else "no_noise"
    mode_str = "QUICK" if args.quick else "FULL"

    print(f"\n  Mode:      {mode_str}")
    print(f"  Noise:     {noise_str}")
    print(f"  Seed:      {args.seed}")
    print(f"  Output:    {plots_dir}/")

    # --- Load data ---
    print("\n  Loading Kaggle dataset...", end=" ")
    data_full = prepare_kaggle_datasets(scenario='extrapolation')
    zeta = data_full['zeta']
    omega_n = data_full['omega_n']
    print("done.")
    print(f"  Physics:   ζ = {zeta:.4f}, ωₙ = {omega_n:.4f}")
    print(f"  Train:     {len(data_full['t_train'])} pts")
    print(f"  Test:      {len(data_full['t_test'])} pts")
    
    # --- Test 1: Data Efficiency ---
    print("\n" + "-" * 70)
    print("  TEST 1: Data Efficiency")
    print("-" * 70)
    print(f"  {'Data':<8} {'Pts':<6} {'Baseline RMSE':<14} {'PINN RMSE':<14} {'Winner':<12}")
    print("-" * 70)

    # Base epochs (scaled by epoch_mult in quick mode)
    scenarios = [
        ("100%", 1.0, int(5000 * epoch_mult), 0.1),
        ("50%", 0.5, int(6000 * epoch_mult), 0.3),
        ("30%", 0.3, int(8000 * epoch_mult), 0.5),
        ("20%", 0.2, int(10000 * epoch_mult), 0.5),
        ("10%", 0.1, int(15000 * epoch_mult), 0.5),
        ("5%", 0.05, int(20000 * epoch_mult), 0.5),
    ]

    results = []
    for name, frac, epochs, lam in tqdm(scenarios, desc="Training scenarios", leave=False):
        data = subsample_training_data(data_full, frac)
        data = add_gaussian_noise(data, args.noise, seed=args.seed)
        n = len(data['t_train'])
        
        model_b, _, m_b = train_and_evaluate('baseline', data, zeta, omega_n, n_epochs=epochs, verbose=False)
        model_p, _, m_p = train_and_evaluate('pinn', data, zeta, omega_n, n_epochs=epochs, lambda_physics=lam, verbose=False)
        
        imp = (m_b['rmse'] - m_p['rmse']) / m_b['rmse'] * 100
        winner = f"PINN +{imp:.0f}%" if imp > 0 else f"Base +{-imp:.0f}%"
        results.append((frac, m_b['rmse'], m_p['rmse'], imp))
        tqdm.write(f"  {name:<8} {n:<6} {m_b['rmse']:<14.4f} {m_p['rmse']:<14.4f} {winner}")

        # Descriptive filename includes noise level
        out_path = plots_dir / f"data_efficiency_{int(frac*100):03d}pct_{n}pts_{noise_str}_{timestamp}.png"
        save_run_plot(
            out_path,
            title=f"Data Efficiency Test: {name} of training data ({n} points)",
            subtitle=f"Training epochs: {epochs} | {noise_str} | seed={args.seed}",
            data_full=data_full,
            data_train=data,
            model_baseline=model_b,
            model_pinn=model_p,
            rmse_baseline=m_b["rmse"],
            rmse_pinn=m_p["rmse"],
            zeta=zeta,
            omega_n=omega_n,
            lambda_physics=lam,
        )
    
    # Find crossover
    print("-" * 70)
    for i in range(len(results) - 1):
        if results[i][3] <= 0 and results[i + 1][3] > 0:
            print(f"  → Crossover between {results[i][0]*100:.0f}% and {results[i+1][0]*100:.0f}% data")
            break
    else:
        pinn_wins = sum(1 for r in results if r[3] > 0)
        print(f"  → PINN wins {pinn_wins}/{len(results)} scenarios")

    # --- Test 2: Extrapolation ---
    print("\n" + "-" * 70)
    print("  TEST 2: Extrapolation Performance")
    print("-" * 70)
    t_train_min = data_full["t_train"].min().item()
    t_train_max = data_full["t_train"].max().item()
    t_test_min = data_full["t_test"].min().item()
    t_test_max = data_full["t_test"].max().item()
    print(f"  Train: t ∈ [{t_train_min:.1f}, {t_train_max:.1f}]")
    print(f"  Test:  t ∈ [{t_test_min:.1f}, {t_test_max:.1f}]")
    extrap_epochs = int(5000 * epoch_mult)
    print(f"\n  Training full models ({extrap_epochs} epochs)...")
    data_full_noisy = add_gaussian_noise(data_full, args.noise, seed=args.seed)
    model_b_full, _, m_b_full = train_and_evaluate(
        'baseline', data_full_noisy, zeta, omega_n, n_epochs=extrap_epochs, verbose=True
    )
    model_p_full, _, m_p_full = train_and_evaluate(
        'pinn', data_full_noisy, zeta, omega_n, n_epochs=extrap_epochs, lambda_physics=0.1, verbose=True
    )

    print(f"\n  Baseline RMSE: {m_b_full['rmse']:.4f}")
    print(f"  PINN RMSE:     {m_p_full['rmse']:.4f}")
    imp_extrap = (m_b_full['rmse'] - m_p_full['rmse']) / m_b_full['rmse'] * 100
    winner_str = "PINN" if imp_extrap > 0 else "Baseline"
    print(f"  → {winner_str} wins by {abs(imp_extrap):.0f}%")

    out_path = plots_dir / f"extrapolation_full_{len(data_full['t_train'])}pts_{noise_str}_{timestamp}.png"
    save_run_plot(
        out_path,
        title=f"Extrapolation Test: Full data ({len(data_full['t_train'])} pts)",
        subtitle=f"Train t∈[{t_train_min:.0f},{t_train_max:.0f}], Test t∈[{t_test_min:.0f},{t_test_max:.0f}] | {noise_str}",
        data_full=data_full,
        data_train=data_full_noisy,
        model_baseline=model_b_full,
        model_pinn=model_p_full,
        rmse_baseline=m_b_full["rmse"],
        rmse_pinn=m_p_full["rmse"],
        zeta=zeta,
        omega_n=omega_n,
        lambda_physics=0.1,
    )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Dataset:  Kaggle damped-harmonic-oscillator")
    print(f"  Physics:  ζ = {zeta:.4f}, ωₙ = {omega_n:.4f}")
    print(f"  Noise:    {noise_str}")
    print(f"  Plots:    {plots_dir}/")
    print("=" * 70)
    print("  ✓ Done\n")


if __name__ == "__main__":
    main()

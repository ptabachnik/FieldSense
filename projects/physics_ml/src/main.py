#!/usr/bin/env python3
"""
PINN vs Baseline NN - Kaggle Damped Harmonic Oscillator Dataset

Run: python3 -m src.main
"""
import torch
import numpy as np
from pathlib import Path

from .data import prepare_kaggle_datasets, subsample_training_data
from .train import train_and_evaluate
 
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
        f"Estimated physics: ζ={zeta:.4f}, ωₙ={omega_n:.4f} | λ_physics={lambda_physics}"
    )
    ax.set_xlabel("time")
    ax.set_ylabel("displacement")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    print("\n" + "="*65)
    print("  PINN vs BASELINE NN - Kaggle Damped Oscillator Dataset")
    print("  x'' + 2ζωₙx' + ωₙ²x = 0")
    print("="*65)
    
    Path("outputs").mkdir(exist_ok=True)
    plots_dir = Path("outputs") / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Kaggle data
    print("\nLoading Kaggle dataset...")
    data_full = prepare_kaggle_datasets(scenario='extrapolation')
    
    zeta = data_full['zeta']
    omega_n = data_full['omega_n']
    
    print(f"Estimated physics: ζ={zeta:.4f}, ω_n={omega_n:.4f}")
    print(f"Training points: {len(data_full['t_train'])}")
    print(f"Test points: {len(data_full['t_test'])}")
    
    # ===== TEST 1: DATA EFFICIENCY =====
    print("\n[TEST 1] Data Efficiency")
    print("-"*65)
    print(f"{'Data':<10} {'Points':<8} {'Baseline':<12} {'PINN':<12} {'Winner':<12}")
    print("-"*65)
    
    results = []
    for name, frac, epochs, lam in [
        ("100%", 1.0, 5000, 0.1),
        ("50%", 0.5, 6000, 0.3),
        ("30%", 0.3, 8000, 0.5),
        ("20%", 0.2, 10000, 0.5),
        ("10%", 0.1, 15000, 0.5),
        ("5%", 0.05, 20000, 0.5),
    ]:
        data = subsample_training_data(data_full, frac)
        n = len(data['t_train'])
        
        model_b, _, m_b = train_and_evaluate('baseline', data, zeta, omega_n, n_epochs=epochs, verbose=False)
        model_p, _, m_p = train_and_evaluate('pinn', data, zeta, omega_n, n_epochs=epochs, lambda_physics=lam, verbose=False)
        
        imp = (m_b['rmse'] - m_p['rmse']) / m_b['rmse'] * 100
        winner = f"PINN +{imp:.0f}%" if imp > 0 else f"Base +{-imp:.0f}%"
        results.append((frac, m_b['rmse'], m_p['rmse'], imp))
        print(f"{name:<10} {n:<8} {m_b['rmse']:<12.4f} {m_p['rmse']:<12.4f} {winner:<12}")

        out_path = plots_dir / f"pinn_vs_baseline_{int(frac*100):03d}pct.png"
        save_run_plot(
            out_path,
            title=f"PINN vs Baseline ({name} train, seed=42, epochs={epochs})",
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
    print("-"*65)
    for i in range(len(results)-1):
        if results[i][3] <= 0 and results[i+1][3] > 0:
            print(f"→ Crossover: between {results[i][0]*100:.0f}% and {results[i+1][0]*100:.0f}% data")
            break
    else:
        # Check overall trend
        pinn_wins = sum(1 for r in results if r[3] > 0)
        print(f"→ PINN wins in {pinn_wins}/{len(results)} scenarios")
    
    print("\n[TEST 2] Extrapolation Performance")
    print("-"*65)
    t_train_min = data_full["t_train"].min().item()
    t_train_max = data_full["t_train"].max().item()
    t_test_min = data_full["t_test"].min().item()
    t_test_max = data_full["t_test"].max().item()
    print(
        "Note: Kaggle 'extrapolation' split includes overlap + extension:\n"
        f"  train t ∈ [{t_train_min:.1f}, {t_train_max:.1f}], "
        f"test t ∈ [{t_test_min:.1f}, {t_test_max:.1f}]"
    )
    
    # Full training
    model_b_full, _, m_b_full = train_and_evaluate('baseline', data_full, zeta, omega_n, n_epochs=5000, verbose=False)
    model_p_full, _, m_p_full = train_and_evaluate('pinn', data_full, zeta, omega_n, n_epochs=5000, lambda_physics=0.1, verbose=False)
    
    print(f"Baseline test RMSE: {m_b_full['rmse']:.4f}")
    print(f"PINN test RMSE:     {m_p_full['rmse']:.4f}")
    imp_extrap = (m_b_full['rmse'] - m_p_full['rmse']) / m_b_full['rmse'] * 100
    print(f"→ {'PINN' if imp_extrap > 0 else 'Baseline'} improvement: {abs(imp_extrap):.0f}%")

    out_path = plots_dir / "pinn_vs_baseline_full_100pct.png"
    save_run_plot(
        out_path,
        title="PINN vs Baseline (100% train, seed=42, epochs=5000)",
        data_full=data_full,
        data_train=data_full,
        model_baseline=model_b_full,
        model_pinn=model_p_full,
        rmse_baseline=m_b_full["rmse"],
        rmse_pinn=m_p_full["rmse"],
        zeta=zeta,
        omega_n=omega_n,
        lambda_physics=0.1,)
    print(f"Saved plots to: {plots_dir}")
    
    print("\n" + "="*65)
    print("  SUMMARY (Kaggle Data)")
    print("="*65)
    print("""
  Dataset: Kaggle damped-harmonic-oscillator
  Physics estimated from data: ζ={:.4f}, ω_n={:.4f}
  """.format(zeta, omega_n))
    print("✓ Done!\n")


if __name__ == "__main__":
    main()

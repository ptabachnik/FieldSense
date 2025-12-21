# PINN vs Baseline NN - Phase 1

Compares Physics-Informed Neural Networks against standard NNs using **Kaggle damped harmonic oscillator dataset**.

## Quick Start

```bash
cd projects/physics_ml
pip install -r requirements.txt
python -m src.main
```

## Results (Kaggle Data)

### Data Efficiency

Numbers below are from the default run (`python -m src.main`, seed=42).

| Data | Points | Baseline RMSE | PINN RMSE | Winner |
|------|--------|--------------:|----------:|--------|
| 100% | 200 | 0.0692 | 0.0724 | Baseline +5% |
| 50% | 100 | 0.2673 | 0.1146 | **PINN +57%** |
| 30% | 60 | 0.1506 | 0.1031 | **PINN +31%** |
| 20% | 40 | 0.2423 | 0.1070 | **PINN +56%** |
| 10% | 20 | 0.1699 | 0.1033 | **PINN +39%** |
| 5% | 10 | 0.1619 | 0.0989 | **PINN +39%** |

**Crossover (seed=42): between 100% and 50% data** — PINN wins below this threshold.

## Key Findings

| Prediction | Result | ✓/✗ |
|------------|--------|-----|
| PINN wins with sparse data | PINN wins at 50% and below (seed=42) | ✓ |
| Baseline wins with abundant data | Baseline wins at 100% (seed=42) | ✓ |

## Physics

Damped harmonic oscillator: `x'' + 2ζωₙx' + ωₙ²x = 0`

- Physics parameters auto-estimated from data: ζ=0.0675, ω_n=1.0443
- **PINN Loss**: `λ_data·MSE(data) + λ_physics·MSE(ODE_residual) + λ_ic·MSE(IC)` (IC optional)

## Dataset

[Kaggle - Damped Harmonic Oscillator](https://www.kaggle.com/datasets/cici118/damped-harmonic-oscillator)

## Project Structure

```
src/
├── main.py     # python -m src.main
├── models.py   # BaselineNN + PINN
├── data.py     # Kaggle data loading
└── train.py    # Training loops
```

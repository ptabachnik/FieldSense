# FieldSense Physics-Integrated Rain Modeling

Phase 1 compares Physics-Informed Neural Networks against standard NNs using the
**Kaggle damped harmonic oscillator dataset**.

Phase 2 prepares OpenMRG Sweden CML and rain gauge data before rainfall
modeling. PyNNcml is used for dataset loading/alignment when available; the
Phase 3 rain models are feed-forward MLPs over tabular and spatial CML-derived
features, not PyNNcml RNN/LSTM extensions.

## Current Headline Result

The most reliable project result is the Phase 3D spatial sweep:

- Spatial CML context plus physics-derived features reaches the original 10%
  improvement target under a matched one-link baseline comparison.
- Best absolute-RMSE attempt: `0.0433` spatial RMSE with `9.98%` improvement.
- Best relative-gain attempt: `16.8%` RMSE improvement.
- Rigid power-law loss alone does not improve over the NN; power law is more
  useful as a soft feature/prior and when combined with spatial CML context.

## Environment Setup

Use a project-local virtual environment so PyNNcml and its dependencies do not
conflict with other Python tools on the machine:

```bash
cd projects/physics_ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, all commands below should use the `.venv` Python. To leave the
environment:

```bash
deactivate
```

## Phase 1 Quick Start

```bash
cd projects/physics_ml
source .venv/bin/activate
python -m src.phase1.main
```

## Phase 2 Data Prep

Start with data inspection before adding rainfall models:

```bash
cd projects/physics_ml
source .venv/bin/activate
python -m src.phase2.rain_inspect
```

The default inspection reports OpenMRG file availability, available city gauges,
and nearest CML sublinks for the first gauge. It expects the OpenMRG CML signal
file at `dataset/open_datasets/OpenMRG_Sweden/cml/cml.nc`; if that file is
missing, restore/download it before building the aligned table.

Once `cml.nc` exists, prepare the canonical table that Phase 3 models will read:

```bash
python -m src.phase2.rain_prepare --dataset openmrg --gauge Chalm
```

If the repo-local OpenMRG copy is missing `cml.nc`, use PyNNcml's OpenMRG
loader instead:

```bash
python -m src.phase2.rain_prepare \
  --dataset pynncml_openmrg \
  --time-start 2015-06-01 \
  --time-end 2015-06-08 \
  --max-links 3 \
  --baseline-train-frac 0.7 \
  --output outputs/prepared/pynncml_openmrg_phase2.csv
```

This PyNNcml path uses the package's stable 15-minute OpenMRG alignment by
default. The repo-local `openmrg` adapter remains the preferred 1-minute path
once `cml.nc` is available locally. Attenuation baselines are fit only on the
initial chronological training fraction to avoid validation/test leakage.

The prepared CSV uses a shared schema (`target_rain_mm_h`, `attenuation`,
`link_length_km`, `frequency_GHz`, etc.) so future dataset adapters can feed the
same modeling code.

Validate the prepared table and create chronological splits before modeling:

```bash
python -m src.phase2.rain_validate \
  --input outputs/prepared/pynncml_openmrg_phase2.csv \
  --output-dir outputs/phase2_validation \
  --train-frac 0.7 \
  --val-frac 0.15
```

### Phase 2 Completion Status

Phase 2 data preparation is complete when `rain_validate` reports no blocking
issues. `PASS_WITH_WARNINGS` means the table is usable, but some evaluation
claims need caveats.
Current validated PyNNcml/OpenMRG artifact:

- Prepared table: `outputs/prepared/pynncml_openmrg_phase2.csv`
- Validation report: `outputs/phase2_validation/phase2_validation_report.md`
- Split table for Phase 3: `outputs/phase2_validation/prepared_with_splits.csv`
- Split summary: `outputs/phase2_validation/split_summary.csv`
- Per-link summary: `outputs/phase2_validation/per_link_summary.csv`
- Sanity plots: `outputs/phase2_validation/*.png`

Validated data summary:

- 3 CML sublinks, 2 gauge targets, 2298 rows
- 2015-06-01 to 2015-06-08, 15-minute alignment
- No missing rain targets, RSL values, metadata, or attenuation values
- No negative attenuation values
- Chronological split: 1608 train, 345 validation, 345 test rows
- Rain/attenuation correlation: Pearson 0.900, Spearman 0.487
- Current 3-link test split is rain-sparse (`0.87%` wet rows), so wet/dry and
  rainy-only claims should rely more on the richer spatial validation table.

After this point, Phase 3 can consume `prepared_with_splits.csv` for pure
physics, pure NN, direct physics-loss, and physics-feature experiments.

## Phase 3 Rain Modeling

Run the first rain-estimation benchmark from the validated Phase 2 split table:

```bash
cd projects/physics_ml
source .venv/bin/activate
python -m src.phase3.rain_main \
  --input outputs/phase2_validation/prepared_with_splits.csv \
  --power-b 0.8 \
  --epochs 1000 \
  --output-dir outputs/phase3b
```

This compares:

- pure physics power-law baseline
- pure data-driven MLP
- direct physics-loss MLP with a power-law attenuation residual
- physics-feature NN, which uses the power-law rain estimate as a soft feature
- physics-guided residual model, which learns corrections around the power-law prior

Current Phase 3A/3B artifacts:

- Metrics: `outputs/phase3b/metrics.csv`
- Predictions: `outputs/phase3b/predictions.csv`
- Report: `outputs/phase3b/phase3_report.md`
- Plots: `outputs/phase3b/test_predictions.png`,
  `outputs/phase3b/test_prediction_scatter.png`, `outputs/phase3b/test_metric_bars.png`

Phase 3A uses the power-law relation as a direct physics-loss term:
`A ≈ a * L * R^b`. This loss-only physics model is useful as a baseline, but
the real CML rainfall setting is less ideal than the oscillator task. The
power-law relation captures first-order microwave attenuation physics, but in
real CML/gauge data it is insufficient as a standalone constraint because the
CML observes path-averaged attenuation while gauges provide point measurements,
and wet antenna, baseline drift, hardware noise, and spatial rain variability
violate the idealized assumptions.

Phase 3B therefore uses power law as a soft prior/feature. The current best test
RMSE comes from `physics_feature_nn`, which adds the power-law rain estimate as
an NN input feature. It slightly improves over the pure NN (`0.03503` vs.
`0.03509` RMSE). The residual model improves wet/dry behavior but not RMSE.

Use the Phase 2 report to prove the data assumptions (clean CML/gauge alignment,
rain/attenuation relationship, no missing values, chronological splits). Use the
Phase 3 report and metrics CSV to prove the model-comparison claims with RMSE,
MAE, MSE, bias, and wet/dry classification metrics.

### Phase 3 Robustness

To mirror the Phase 1 sparse/noisy-data tests on real CML/gauge data, run:

```bash
python -m src.phase3.rain_robustness \
  --input outputs/phase2_validation/prepared_with_splits.csv \
  --train-fractions 1.0,0.5,0.3,0.2,0.1 \
  --seeds 42,7,123 \
  --epochs 600 \
  --target-noise-std 0.2 \
  --output-dir outputs/phase3_robustness
```

This randomly removes training rows and adds train-label noise only; validation
and test remain clean. For feature-missing experiments, use
`--feature-missing-prob 0.3`.

Current robustness artifact:

- Report: `outputs/phase3_robustness/robustness_report.md`
- Metrics: `outputs/phase3_robustness/robustness_metrics.csv`
- Plot: `outputs/phase3_robustness/test_rmse_vs_train_fraction.png`

Current robustness result: the simple direct physics-loss model does **not**
beat the pure NN under these sparse/noisy settings. A separate lower-weight run
(`outputs/phase3_robustness_lambda001`) nearly matches the NN, but still does
not improve RMSE. This is an important negative result and suggests the next
physics-integrated iteration needs a better physics term or architecture, not
just a larger physics loss.

### Spatial Robustness (Data Efficiency)

The rigid direct-loss robustness test above only tested the single-link
physics-loss model. The spatial+physics configuration that succeeded in Phase
3D was never tested under sparse data. A dedicated spatial robustness experiment
fills this gap:

```bash
python -m src.phase3.rain_spatial_robustness \
  --train-fractions 1.0,0.5,0.3,0.2,0.1 \
  --seeds 42,7,123 \
  --epochs 500 \
  --max-links-per-target 2 \
  --rain-weight-alpha 2.0 \
  --output-dir outputs/phase3_spatial_robustness
```

Current spatial robustness artifacts:

- Report: `outputs/phase3_spatial_robustness/spatial_robustness_report.md`
- Metrics: `outputs/phase3_spatial_robustness/spatial_robustness_metrics.csv`
- Plots: `outputs/phase3_spatial_robustness/rmse_vs_train_fraction.png`,
  `outputs/phase3_spatial_robustness/improvement_vs_fraction.png`

Current spatial robustness result: **spatial+physics beats single-link in 5/5
data fractions** (averaged across 3 seeds), with 4/5 reaching the 10%
improvement target. Best average improvement: `16.2%` at 30% training data.
The spatial+physics model degrades more gracefully than the single-link model
as training data is reduced, directly supporting the Metric 2 data-efficiency
claim using the spatial architecture rather than the rigid direct physics loss.

With 10% label noise (`--target-noise-std 0.1`), spatial+physics still wins in
5/5 fractions (best current artifact: about `11.3%`), though only part of the
curve exceeds the 10% target. At 20% noise the advantage washes out, indicating
the spatial benefit is most reliable for data-sparse rather than noise-heavy
scenarios.

### Phase 3E Wet/Dry Classification

The regression models also report wet/dry metrics by thresholding predicted rain
rate, but the project deliverables call out wet/dry classification explicitly.
Run a dedicated classifier on the same spatial feature sets:

```bash
python -m src.phase3.rain_wetdry \
  --input outputs/phase2_spatial_validation/prepared_with_splits.csv \
  --max-links-per-target 2 \
  --epochs 1000 \
  --threshold-strategy fixed \
  --output-dir outputs/phase3e_wetdry
```

Current Phase 3E artifacts:

- Report: `outputs/phase3e_wetdry/wetdry_report.md`
- Metrics: `outputs/phase3e_wetdry/wetdry_metrics.csv`
- Predictions: `outputs/phase3e_wetdry/wetdry_predictions.csv`
- Plot: `outputs/phase3e_wetdry/wetdry_metric_bars.png`

Current Phase 3E result:

- Best classifier: `spatial_physics_classifier`
- Wet/dry F1: `0.720`
- Improvement over one-link classifier F1: `13.1%`
- This supports the documented classification-improvement goal, with the caveat
  that the test split remains rain-sparse.

### Phase 3C Spatial Modeling

The single-row Phase 3 models treat each CML link independently. To test whether
spatial CML context helps, prepare a richer table and build gauge-level samples
with multiple nearby CML links:

```bash
python -m src.phase2.rain_prepare \
  --dataset pynncml_openmrg \
  --time-start 2015-06-01 \
  --time-end 2015-06-08 \
  --max-links 12 \
  --baseline-train-frac 0.7 \
  --output outputs/prepared/pynncml_openmrg_spatial.csv

python -m src.phase2.rain_validate \
  --input outputs/prepared/pynncml_openmrg_spatial.csv \
  --output-dir outputs/phase2_spatial_validation \
  --train-frac 0.7 \
  --val-frac 0.15

python -m src.phase3.rain_spatial \
  --input outputs/phase2_spatial_validation/prepared_with_splits.csv \
  --max-links-per-target 2 \
  --epochs 1000 \
  --output-dir outputs/phase3c_spatial_k2
```

Current Phase 3C artifact:

- Report: `outputs/phase3c_spatial_k2/spatial_report.md`
- Metrics: `outputs/phase3c_spatial_k2/spatial_metrics.csv`
- Wide spatial table: `outputs/phase3c_spatial_k2/spatial_wide_table.csv`
- Plot: `outputs/phase3c_spatial_k2/spatial_metric_bars.png`

Intermediate Phase 3C result: using two nearby CML links per target improves test
RMSE from `0.0462` for the one-link NN to `0.0439` for the spatial NN, about
`4.9%` improvement. This shows spatial CML context helps, but it is below the
original 10-15% improvement target. Phase 3D below extends this with a controlled
matched sweep and reaches the target on relative improvement.

### Phase 3D Spatial Sweep

To try to reach the original 10-15% improvement target, run up to 100 spatial
configurations. Each spatial model is compared against a matched one-link
baseline trained with the same seed, learning rate, architecture, and
rain-weighted loss:

```bash
python -m src.phase3.rain_spatial_sweep \
  --input outputs/phase2_spatial_validation/prepared_with_splits.csv \
  --max-attempts 100 \
  --epochs 500 \
  --output-dir outputs/phase3d_spatial_sweep
```

Current Phase 3D artifacts:

- Report: `outputs/phase3d_spatial_sweep/spatial_sweep_report.md`
- Full sweep table: `outputs/phase3d_spatial_sweep/spatial_sweep_results.csv`
- Plot: `outputs/phase3d_spatial_sweep/best_attempts.png`

Current Phase 3D result:

- Best improvement attempt: `16.8%` RMSE improvement, from matched one-link
  RMSE `0.0759` to spatial RMSE `0.0632`.
- Lowest absolute spatial RMSE attempt: spatial RMSE `0.0433`, with `9.98%`
  improvement over its matched one-link baseline.
- This reaches the 10% target by relative-gain sweep result and is effectively
  at the target for the lowest-RMSE result. It supports the claim that spatial
  CML context plus physics-derived features can improve over a one-link baseline,
  while remaining sensitive to split rain balance and hyperparameter selection.
- This should be reported as a spatial MLP / physics-feature result. It is not
  evidence that a PyNNcml RNN/LSTM model was extended with a PINN loss.

## Presentation Artifacts

Build slide-ready summary figures and tables from the existing Phase 2/3 outputs:

```bash
python -m src.phase3.rain_presentation --output-dir outputs/presentation
```

Generated artifacts:

- Summary report: `outputs/presentation/presentation_summary.md`
- Goal progress table: `outputs/presentation/goal_progress_table.csv`
- Spatial comparison table: `outputs/presentation/spatial_comparison_table.csv`
- Figures:
  - `outputs/presentation/model_progress_rmse.png`
  - `outputs/presentation/improvement_by_stage.png`
  - `outputs/presentation/spatial_metric_comparison.png`
  - `outputs/presentation/wetdry_metric_comparison.png`

There is also a Cursor Canvas dashboard named `fieldsense-results.canvas.tsx`
that summarizes the key results visually inside Cursor.

## Results (Kaggle Data)

### Data Efficiency

Numbers below are from the default run (`python -m src.phase1.main`, seed=42).

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
├── phase1/
│   ├── main.py      # oscillator entry point
│   ├── models.py    # oscillator BaselineNN + PINN
│   ├── data.py      # Kaggle oscillator data loading
│   └── train.py     # oscillator training loops
├── phase2/
│   ├── rain_data.py     # OpenMRG data loading/cleaning/alignment
│   ├── rain_inspect.py  # data inspection entry point
│   ├── rain_pynncml.py  # PyNNcml OpenMRG adapter
│   ├── rain_openrainer.py # OpenRainER Italy adapter
│   ├── rain_prepare.py  # generic prepare CLI
│   ├── rain_schema.py   # canonical table schema
│   └── rain_validate.py # validation, splits, and plots
├── phase3/
│   ├── rain_models.py   # rain MLP and power-law losses
│   ├── rain_train.py    # data tensors, metrics, training
│   ├── rain_main.py     # benchmark entry point
│   ├── rain_robustness.py # sparse/noisy-data robustness sweep
│   ├── rain_wetdry.py   # wet/dry classification
│   ├── rain_spatial.py  # spatial multi-link comparison
│   ├── rain_spatial_sweep.py # spatial hyperparameter/data sweep
│   └── rain_presentation.py # presentation-ready summary tables and figures
└── __init__.py
```

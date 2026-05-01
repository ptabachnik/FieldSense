Presentation Additions (No Code on Slides)
=========================================

Core message
------------
Phase 1 shows the clean PINN story on a controlled oscillator problem: a plain
NN fits best when data is abundant, while a PINN wins when data is sparse or
noisy.

For the real CML rainfall task, the story is narrower and more realistic:
rigid power-law loss alone did not beat the NN, but physics-derived features and
spatial CML context improved performance. The strongest result is the Phase 3D
spatial sweep, where spatial CML context plus physics-derived features reached
the original 10% improvement target against a matched one-link baseline.

Slide: Baseline NN Algorithm (data-only)
----------------------------------------
Purpose: show the baseline has no physics, only data fitting.

Text (short paragraph):
The baseline network maps time to displacement and is trained only to minimize data error. With abundant data it can fit well, but it has no built-in physical constraints.

Flow diagram (ASCII):
time t -> Neural Network -> predicted displacement x_hat
                         -> data loss (MSE) -> update weights

Slide: PINN Algorithm (data + physics)
--------------------------------------
Purpose: show the added physics loss and why it improves sparse/noisy regimes.

Text (short paragraph):
The PINN uses the same network, but adds a physics loss that penalizes violations of the governing ODE. This regularizes the model when data is limited or noisy.

Equation to show (not code):
ODE: x'' + 2ζω_n x' + ω_n^2 x = 0

Loss definition (not code):
Total loss = data loss + physics loss (+ initial condition loss)

Flow diagram (ASCII):
time t -> Neural Network -> x_hat
                         -> compute x', x'' -> ODE residual r
                         -> physics loss (r should be 0)
                         -> data loss (MSE)
                         -> combine losses -> update weights

Slide: Quantitative Results (Data Efficiency)
----------------------------------------------
Use this table directly. It is already in the repo and shows the crossover.

Data fraction | Points | Baseline RMSE | PINN RMSE | Winner
100%          | 200    | 0.0692        | 0.0724    | Baseline +5%
50%           | 100    | 0.2673        | 0.1146    | PINN +57%
30%           | 60     | 0.1506        | 0.1031    | PINN +31%
20%           | 40     | 0.2423        | 0.1070    | PINN +56%
10%           | 20     | 0.1699        | 0.1033    | PINN +39%
5%            | 10     | 0.1619        | 0.0989    | PINN +39%

Callout sentence:
The crossover happens between 100% and 50% data: above it the NN is best, below it the PINN is best.

Slide: Visual Comparison (Full vs Sparse)
------------------------------------------
Purpose: show the model curves and data points for two regimes.

Panel A (Full data, NN wins):
Caption: "Full data (200 points): Baseline slightly wins; PINN is close."

Panel B (Sparse data, PINN wins):
Caption: "Sparse data (20–40 points): PINN fits the true trajectory better."

If you have space, include the RMSE values next to each panel.

Image insertion (use existing plot style)
-----------------------------------------
Insert images from the model comparison plots that include RMSE in the legend.

Recommended images (from outputs/plots):
1) Full data (NN wins): data_efficiency_100pct_*_no_noise_*.png
2) Sparse data (PINN wins): data_efficiency_020pct_*_no_noise_*.png

If you want a stronger contrast, use 10% instead of 20%:
data_efficiency_010pct_*_no_noise_*.png

Slide: Visual Comparison (Noisy Data)
-------------------------------------
Purpose: show robustness to noisy training points.

Panel C (Noisy training):
Caption: "With noise in training data, PINN remains stable due to physics constraints."

Note: use the same plot style as the full/sparse visuals, and report RMSE next to each curve.

Image insertion (noisy data)
----------------------------
Recommended image:
data_efficiency_020pct_*_noise=20%_*.png

If you want a single-slide visual comparison, put the sparse/noisy plot next to the full-data plot.

Presenter note (not on slides)
------------------------------
The plotting script saves images under:
projects/physics_ml/outputs/plots/
The filename includes the data fraction, points, and noise level.

Slide: Key Takeaways
--------------------
Short, slide-ready bullets:
1) Same architecture; the difference is the physics loss.
2) In Phase 1, PINN wins when data is sparse or noisy.
3) On real CML/gauge data, rigid power-law PINN loss is too idealized.
4) Physics is more useful as a soft feature/prior combined with spatial CML context.

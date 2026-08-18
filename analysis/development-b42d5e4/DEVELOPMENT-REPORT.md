# Development-only analysis

This report is **provisional and non-confirmatory**. It uses only the twelve authorized development trajectories.

Validated 492/492 raw logit tensors (SHA-256, float32, shape 5000x10) and modeled 480 checkpoints (epochs 5-200).

## Nested grouped cross-validation

Outer folds leave one complete seed trajectory out (12 folds); inner three-fold GroupKFold keeps trajectories intact. Behavioral PCA is randomized, fitted only on each training fold, and capped at 32 components; component prefixes 8/16/32 and Ridge alpha 0.01/0.1/1/10/100 are selected by inner MAE.

| model | MAE | R^2 | Spearman | delta MAE vs all-nuisance |
|---|---:|---:|---:|---:|
| epoch_only | 0.17966 | 0.1465 | 0.2884 | +0.00000 |
| train_loss_accuracy | 0.15181 | 0.4663 | -0.4271 | +0.00000 |
| probe_nuisance | 0.11745 | 0.5872 | 0.4330 | +0.00000 |
| all_nuisance | 0.11936 | 0.5704 | 0.4276 | +0.00000 |
| evaluation_accuracy_control | 0.17559 | 0.1949 | -0.1404 | +0.00000 |
| behavioral_raw_logits | 0.14018 | 0.4991 | 0.4651 | -0.02082 |
| behavioral_class_centered | 0.14019 | 0.4992 | 0.4654 | -0.02084 |
| behavioral_scale_normalized | 0.11938 | 0.6595 | 0.6959 | -0.00003 |
| behavioral_probabilities | 0.10939 | 0.6997 | 0.8418 | +0.00997 |

## Whole-run uncertainty

Probability behavioral improvement over all-nuisance: mean paired MAE improvement 0.00997; improved 7/12 runs; fixed-seed 10,000-bootstrap 95% percentile CI [-0.03670, 0.05678].
Run influence is substantial: the largest improvement is +0.21537 (run-ef7523a1358a/seed-881886696) and the largest deterioration is -0.16637 (run-be83c9d3e5af/seed-2325060526).
Trajectory-identity permutation (10,000 reassignments, complete paths and epoch order preserved): behavioral improvement one-sided p=0.03120; all-nuisance identity-control p=0.04140; behavioral identity p=0.00120.

## Hyperparameters selected on all development runs (lock before confirmation)

- epoch_only: alpha=100.0
- train_loss_accuracy: alpha=0.01
- probe_nuisance: alpha=10.0
- all_nuisance: alpha=1.0
- evaluation_accuracy_control: alpha=0.01
- behavioral_probabilities: representation=probabilities, components=16, alpha=1.0

Selected development representation: **probabilities** (smallest outer-CV behavioral MAE).

## Development-fit geometry (exploratory)

Cross-run distance vs |g difference| Spearman: Euclidean 0.2298; cosine -0.1168.
Matched-epoch subset: Euclidean 0.0162; cosine 0.1298.
Accuracy-binned (width 0.02): Euclidean -0.1576; cosine -0.0349.
Mean Euclidean state distances: g=0 to g=0 5.7357; g>0 to g>0 5.5265; g=0 to g>0 7.0590.
Adjacent-checkpoint direction alignment (exploratory): mean cosine 0.0514.

## Development decision

The probability representation contains a promising development-only signal for predicting the overfitting gap beyond the recorded nuisance summaries: it has the best pooled held-out-trajectory MAE and strong pooled rank correlation. This is not yet a stable effect: only 7/12 held-out trajectories improve, the whole-run bootstrap interval crosses zero, and two trajectories have large opposing influence. The identity permutation is a reason to carry the locked model into confirmation, not a substitute for confirmation.

The stronger claim of a coherent overfitting geometry is not supported by this development set. Distance-to-|g difference| associations are near zero or negative after matching epoch or binning evaluation accuracy, and adjacent trajectory directions are almost orthogonal on average. The larger between-state mean distance is at most a coarse state-separation hint, not evidence of a shared continuous path.

Lock probabilities with 16 PCA components and Ridge alpha 1.0 for confirmation. The selected representation, hyperparameters, and geometry are development decisions only; none is confirmation evidence.

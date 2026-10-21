# Evaluation and figures

```python
from stochact import Evaluator
from stochact.visualization import plot_reliability, plot_risk_coverage

evaluator = Evaluator(model, task='classification', n_samples=5)
report = evaluator.evaluate(test_loader, return_details=True)
print(report.metrics)
plot_reliability(report, save_path='calibration.pdf')
plot_risk_coverage(report, save_path='risk.png')
```

Classification: accuracy, macro precision/recall/F1, weighted F1, NLL, ECE, Brier.

Segmentation: pixel accuracy, foreground Dice/IoU, precision/recall, macro foreground metrics, calibration, correlation of uncertainty and pixel error. Binary 2D segmentation additionally provides tolerance-based correctness/completeness/quality (not a benchmark-specific centerline protocol).

Regression: MAE, MSE, RMSE. Out-of-distribution comparison via `evaluator.evaluate_ood(id_loader, ood_loader, score='variance')` gives AUROC, AUPR and FPR95. Available OOD scores: variance, entropy, mutual information, negative max probability.

Plots: `plot_training`, `plot_uncertainty`, `plot_reliability`, `plot_risk_coverage`, `plot_ood`, `plot_mc_convergence`. Install `.[plots]` or `.[all]` to get Matplotlib. Every plotting function returns a Matplotlib figure and axes for further customization.

For segmentation, the uncertainty heatmap is predictive variance across stochastic passes, **not** a calibrated probability of a segmentation error. Entropy and disagreement scores are not necessarily strictly epistemic or aleatoric quantities.

"""Optional plotting utilities; install `stochact[plots]` for Matplotlib."""
from .plots import (plot_training, plot_uncertainty, plot_reliability,
                    plot_risk_coverage, plot_ood, plot_mc_convergence)

__all__ = ["plot_training", "plot_uncertainty", "plot_reliability", "plot_risk_coverage", "plot_ood", "plot_mc_convergence"]

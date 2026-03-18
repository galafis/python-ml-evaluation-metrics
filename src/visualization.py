"""Metrics visualization module.

Plot confusion matrices, ROC curves, and metric comparisons.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class MetricsVisualizer:
    """Visualize ML evaluation metrics."""

    def __init__(self, figsize: Tuple[int, int] = (10, 8), style: str = "whitegrid"):
        """Initialize visualizer."""
        self.figsize = figsize
        sns.set_style(style)

    def plot_confusion_matrix(
        self,
        cm: List[List[int]],
        labels: Optional[List[str]] = None,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot confusion matrix heatmap."""
        cm_array = np.array(cm)
        if normalize:
            cm_array = cm_array.astype("float") / cm_array.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            cm_array, annot=True, fmt=".2f" if normalize else "d",
            cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_title(title)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_roc_curve(
        self,
        fpr: List[float],
        tpr: List[float],
        auc_score: float,
        title: str = "ROC Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"ROC curve saved to {save_path}")

        return fig

    def plot_metrics_comparison(
        self,
        metrics: Dict[str, float],
        title: str = "Model Metrics Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot bar chart of metrics."""
        fig, ax = plt.subplots(figsize=self.figsize)
        names = list(metrics.keys())
        values = list(metrics.values())
        colors = sns.color_palette("husl", len(names))

        bars = ax.bar(names, values, color=colors)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.15)
        ax.set_ylabel("Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_precision_recall_curve(
        self,
        precision: List[float],
        recall: List[float],
        ap_score: float,
        title: str = "Precision-Recall Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot Precision-Recall curve."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AP = {ap_score:.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve, ConfusionMatrixDisplay,
)
from sklearn.metrics import confusion_matrix

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def heatmap(df):
    hour_dow = (
        df.groupby(["hour", "day_of_week"])["accident"]
        .mean()
        .reset_index()
    )

    pivot = hour_dow.pivot(
        index="hour",
        columns="day_of_week",
        values="accident"
    )

    pivot.columns = days

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd"
    )

    plt.title("Mean Accident Rate by Hour and Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Hour")

    plt.tight_layout()
    plt.show()

def plot_accidents_time_dimensions(df):
    colors = [
        "#5B84B1",
        "#7AA095",
        "#A68A64",
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5),
        sharey=True
    )

    # Time Period
    time_period_order = [
        "night",
        "morning",
        "afternoon",
        "evening"
    ]

    time_period_stats = (
        df.groupby("time_period")["accident"]
        .mean()
        .reindex(time_period_order)
        .reset_index()
    )

    ax = sns.barplot(
        data=time_period_stats,
        x="time_period",
        y="accident",
        ax=axes[0],
        color=colors[0]
    )

    axes[0].set_title("By Time Period")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean Accident Rate")

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f%%",
            labels=[
                f"{v * 100:.2f}%"
                for v in time_period_stats["accident"]
            ],
            padding=3
        )

    # Day of Week
    dow_stats = (
        df.groupby("day_of_week")["accident"]
        .mean()
        .reset_index()
    )

    ax = sns.barplot(
        data=dow_stats,
        x="day_of_week",
        y="accident",
        ax=axes[1],
        color=colors[1]
    )

    axes[1].set_title("By Day of Week")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(days)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f%%",
            labels=[
                f"{v * 100:.2f}%"
                for v in dow_stats["accident"]
            ],
            padding=3
        )

    # Month
    month_stats = (
        df.groupby("month")["accident"]
        .mean()
        .reset_index()
    )

    ax = sns.barplot(
        data=month_stats,
        x="month",
        y="accident",
        ax=axes[2],
        color=colors[2]
    )

    axes[2].set_title("By Month")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")

    axes[2].set_xticks(range(12))
    axes[2].set_xticklabels(months)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f%%",
            labels=[
                f"{v * 100:.2f}%"
                for v in month_stats["accident"]
            ],
            padding=3,
            fontsize=8
        )

    # Formatting
    for ax in axes:
        ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1)
        )

    fig.suptitle(
        "Accident Rate Across Time Dimensions",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

def plot_accidents_traffic(df):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharey=True
    )

    # Congestion
    df_plot = df.copy()

    df_plot["congestion_bin"] = pd.qcut(
        df_plot["avg_congestion"],
        q=10,
        duplicates="drop"
    )

    congestion_stats = (
        df_plot
        .groupby("congestion_bin", observed=True)
        .agg(
            avg_congestion=("avg_congestion", "mean"),
            accident_rate=("accident", "mean")
        )
        .reset_index()
    )

    sns.lineplot(
        data=congestion_stats,
        x="avg_congestion",
        y="accident_rate",
        marker="o",
        color="#7AA095",
        ax=axes[0]
    )

    for x, y in zip(
            congestion_stats["avg_congestion"],
            congestion_stats["accident_rate"]
    ):
        axes[0].annotate(
            f"{y * 100:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8
        )

    axes[0].set_title(
        "Mean Accident Rate vs Congestion"
    )

    axes[0].set_xlabel(
        "Average Congestion"
    )

    axes[0].set_ylabel(
        "Mean Accident Rate"
    )

    # Speed Ratio
    df_plot["speed_bin"] = pd.qcut(
        df_plot["speed_ratio"],
        q=10,
        duplicates="drop"
    )

    speed_stats = (
        df_plot
        .groupby("speed_bin", observed=True)
        .agg(
            speed_ratio=("speed_ratio", "mean"),
            accident_rate=("accident", "mean")
        )
        .reset_index()
    )

    sns.lineplot(
        data=speed_stats,
        x="speed_ratio",
        y="accident_rate",
        marker="o",
        color="#5B84B1",
        ax=axes[1]
    )

    for x, y in zip(
            speed_stats["speed_ratio"],
            speed_stats["accident_rate"]
    ):
        axes[1].annotate(
            f"{y * 100:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8
        )

    axes[1].set_title(
        "Mean Accident Rate vs Speed Ratio"
    )

    axes[1].set_xlabel(
        "Speed Ratio"
    )

    axes[1].set_ylabel("")

    # Formatting
    for ax in axes:
        ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1)
        )
        ax.grid(
            alpha=0.3
        )

    fig.suptitle(
        "Impact of Traffic Conditions on Accident Rate",
        fontsize=15
    )

    plt.tight_layout()

    plt.show()

def plot_roc_curve(
    y_true,
    y_proba,
    model_name,
    auc,
    save_path=None,
):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_proba
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def plot_pr_curve(
    y_true,
    y_proba,
    model_name,
    auc,
    save_path=None,
):

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            y_proba
        )
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"{model_name} (AUC = {auc:.3f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")

    plt.grid()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def plot_feature_importance(
    feature_importances,
    features,
    top_n=15
):

    importance_df = pd.DataFrame(
        {
            "feature": features,
            "importance": feature_importances
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.title(
        f"Top {top_n} Feature Importances"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_mlp_loss_curve(
    pipeline,
    title="MLP Training Loss",
    save_path=None,
):
    loss_curve = (
        pipeline.named_steps["model"]
        .loss_curve_
    )

    plt.figure(figsize=(8, 5))

    plt.plot(loss_curve)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(title)

    plt.grid(True)

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

def plot_mlp_validation_curve(
    pipeline,
    title="MLP Validation Score",
    save_path=None,
):
    validation_scores = (
        pipeline.named_steps["model"]
        .validation_scores_
    )

    plt.figure(figsize=(8, 5))

    plt.plot(validation_scores)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Score")

    plt.title(title)

    plt.grid(True)

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

def plot_threshold_metrics(
    threshold_results,
):
    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        threshold_results["threshold"],
        threshold_results["precision"],
        label="Precision",
    )

    plt.plot(
        threshold_results["threshold"],
        threshold_results["recall"],
        label="Recall",
    )

    plt.plot(
        threshold_results["threshold"],
        threshold_results["f1"],
        label="F1",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Score")

    plt.title(
        "Metrics vs Threshold"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.show()

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name,
):
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    tick_labels = ["No Accident", "Accident"]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.tight_layout()
    plt.show()

def common_conf_matrix(cm_df):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 10)
    )

    axes = axes.flatten()

    for idx, (_, row) in enumerate(cm_df.iterrows()):

        cm = np.array(
            [
                [row["tn"], row["fp"]],
                [row["fn"], row["tp"]],
            ],
            dtype=float,
        )

        cm = (
                cm
                / cm.sum(axis=1, keepdims=True)
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "No Accident",
                "Accident",
            ],
        )

        disp.plot(
            ax=axes[idx],
            colorbar=False,
            cmap="Blues",
        )

        axes[idx].set_title(
            row["model"]
        )

    for idx in range(
            len(cm_df),
            len(axes)
    ):
        fig.delaxes(
            axes[idx]
        )

    plt.tight_layout()

    plt.show()
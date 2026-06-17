import pandas as pd
import seaborn as sns
import ptitprince as pt

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

def make_autopct(counts):
    """Convert categorical data to form N= % to display in pie chart"""
    def autopct(pct):
        # Keep a static counter across calls
        nonlocal idx
        value = counts.tolist()[idx]
        result = f'N={value}\n ({pct:.1f}%)'
        idx += 1
        return result
    idx = 0
    return autopct

def custom_pie(counts, alpha = 0.7):
    # Create figure and axes
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
    fig, axes = plt.subplots(figsize=(18, 4))
    fig.subplots_adjust(right=0.85)  # leave space for the legend
    pie_colors = [to_rgba(c, alpha=alpha) for c in base_colors[:len(counts)]]
    axes.pie(counts, 
                labels=None, 
                # autopct=lambda pct: f'N = {int(pct/100.*counts[counts.columns[i]].sum())}\n ({pct:.1f}%)', 
                autopct=make_autopct(counts[counts.columns[i]]),
                colors=pie_colors,
                counterclock=False,
                startangle=90)
    # axes[i].set_title(f"{df_severity_converted.columns[i+1]} visit\n Total: {np.sum(counts[counts.columns[i]])}")
    # plt.legend(severity_order, loc='center left', bbox_to_anchor=(0.92, 0.5))

def raincloud_plot(y_test, y_test_pred_proba, fontsize=15, ClassLabel=["Non-OSA", "OSA"]):
    df = pd.DataFrame({
        "TrueLabel": list(y_test.values.astype(int).astype(str)),     # Ensure categorical type for grouping
        "PredictedProb": y_test_pred_proba.astype(float)            # Ensure numeric type for plotting
    })
    df["TrueLabel"] = df["TrueLabel"].astype(str).map({"0": ClassLabel[0], "1": ClassLabel[1]})  # categorical for grouping

    # Set seaborn style
    sns.set(style="whitegrid")

    # Vertical raincloud plot
    fig, ax = plt.subplots(figsize=(10, 5))
    pt.RainCloud(
        x="TrueLabel",       # categorical
        y="PredictedProb",   # numeric
        data=df,
        palette="Set2",
        bw=0.2,
        width_viol=0.6,
        alpha=0.65,
        orient='h',         # vertical orientation works reliably
        ax=ax,
        order=[ClassLabel[1], ClassLabel[0]]    # order of categories, 1 always on top
        # pointplot=True,

    )

    # Scale probability from 0 to 1
    ax.set_xlim(0, 1)

    # Labels and title
    ax.set_xlabel("Predicted Probability", fontsize=fontsize)
    ax.set_ylabel("True Class", fontsize=fontsize)

    return fig, ax
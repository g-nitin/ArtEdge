import argparse
import os
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_publication_style():
    """Set publication-ready parameters for plots"""
    plt.rcParams.update(
        {
            "font.family": "Arial",  # Common in scientific publications
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "figure.figsize": (7, 5),  # Common figure size in publications
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
        }
    )


def load_and_process_data(csv_path):
    """
    Load raw performance data from CSV and calculate standard errors

    Expected CSV format:
    Model,Platform,Run1,Run2,Run3,...
    AdaIN,Laptop,670,683,665,...
    """
    # Load the raw data
    df_raw = pd.read_csv(csv_path)

    # Identify columns with run data (assuming they start with 'Run')
    run_cols = [col for col in df_raw.columns if col.startswith("Run")]

    if not run_cols:
        # If there are no 'Run' columns, assume all columns except Model and Platform are run data
        id_cols = ["Model", "Platform"]
        run_cols = [col for col in df_raw.columns if col not in id_cols]

    # Calculate mean and standard error for each model/platform combination
    results = []
    for _, row in df_raw.iterrows():
        model = row["Model"]
        platform = row["Platform"]

        # Get all performance values for this configuration
        performance_values = [row[col] for col in run_cols]

        # Calculate mean and standard error
        mean_performance = np.mean(performance_values)
        std_error = np.std(performance_values, ddof=1) / np.sqrt(
            len(performance_values)
        )

        results.append(
            {
                "Model": model,
                "Platform": platform,
                "Performance": mean_performance,
                "StdError": std_error,
                "NumSamples": len(performance_values),
            }
        )

    # Create a DataFrame with the results
    processed_df = pd.DataFrame(results)

    return processed_df


def plot_grouped_bar(
    df, log_scale=False, save_path="figure.png", show_values=True, error_bars=True
):
    """
    Create a publication-ready grouped bar chart

    Parameters:
    -----------
    df : pandas DataFrame
        Data to plot
    log_scale : bool
        Whether to use log scale for y-axis
    save_path : str
        Path to save the figure
    show_values : bool
        Whether to show values above bars
    error_bars : bool
        Whether to show error bars
    """
    # Color-blind friendly palette with high contrast
    colors = {
        "Laptop": "#0173B2",  # Blue
        "Mobile": "#DE8F05",  # Orange/amber
    }

    fig = plt.figure(figsize=(7, 5))

    # Create proper margins
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.92, bottom=0.15, left=0.12, right=0.92)
    ax = plt.subplot(gs[0, 0])

    # Plot with error bars
    if error_bars:
        # Create custom error bars
        bar_plot = sns.barplot(
            data=df,
            x="Model",
            y="Performance",
            hue="Platform",
            palette=colors,
            ax=ax,
            errorbar=None,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
        )

        # Add error bars manually
        # We need to get the x position for each bar
        x_coords = []
        heights = []
        errors = []
        platforms = []

        for i, model in enumerate(df["Model"].unique()):
            for j, platform in enumerate(df["Platform"].unique()):
                mask = (df["Model"] == model) & (df["Platform"] == platform)
                if mask.any():
                    row = df[mask].iloc[0]
                    # Calculate position (depends on the number of hue categories)
                    width = 0.8 / len(df["Platform"].unique())
                    pos = i + (j - 0.5 * (len(df["Platform"].unique()) - 1)) * width
                    x_coords.append(pos)
                    heights.append(row["Performance"])
                    errors.append(row["StdError"])
                    platforms.append(platform)

        # Plot error bars
        ax.errorbar(
            x=x_coords,
            y=heights,
            yerr=errors,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3,
            capthick=1,
        )
    else:
        bar_plot = sns.barplot(
            data=df,
            x="Model",
            y="Performance",
            hue="Platform",
            palette=colors,
            ax=ax,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
        )

    # Optional: log scale with appropriate tick formatting
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Performance (ms, log scale)")
        # Format y-axis labels with scientific notation for large numbers
        from matplotlib.ticker import ScalarFormatter

        ax.yaxis.set_major_formatter(ScalarFormatter())
    else:
        ax.set_ylabel("Performance (ms)")

    # Annotations - show values on top of bars if requested
    if show_values:
        for p in bar_plot.patches:
            height = p.get_height()
            if height > 1000:
                value_text = f"{height / 1000:.1f}k"  # Display as k for thousands
            else:
                value_text = f"{height:.1f}"

            ax.annotate(
                value_text,
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                fontweight="normal",
                xytext=(0, 3),
                textcoords="offset points",
            )

    # Set title with common scientific formatting
    ax.set_title(
        "Model Performance Comparison: Laptop vs Mobile", fontweight="bold", pad=15
    )

    # Customize x-axis
    ax.set_xlabel("Model Architecture")

    # Customize legend
    legend = ax.legend(
        title="Platform",
        frameon=True,
        edgecolor="black",
        framealpha=1,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    legend.get_frame().set_linewidth(0.8)

    # Add subtle grid for y-axis only
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)  # Place gridlines behind bars

    # Set tick parameters
    ax.tick_params(axis="both", which="major", length=4, width=0.8)

    # Add subtle box around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Tighten layout and save
    # plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600)

    # Return the figure and axis for potential further customization
    return fig, ax


def create_paper_figures(df, output_dir):
    """Create all figures needed for the publication"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Standard plot with error bars
    plot_grouped_bar(
        df, log_scale=False, save_path=output_dir / "figure1_standard_scale.png"
    )

    # Log scale plot for better visualization of differences
    plot_grouped_bar(df, log_scale=True, save_path=output_dir / "figure2_log_scale.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready performance charts from raw data"
    )
    parser.add_argument(
        "input_csv", help="Path to the CSV file with raw performance data"
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__)),
        help="Directory to save generated figures",
    )

    args = parser.parse_args()

    # Set the plotting style
    set_publication_style()

    # Load and process the data
    df = load_and_process_data(args.input_csv)

    # Print the calculated statistics
    print("Calculated performance statistics:")
    for _, row in df.iterrows():
        print(
            f"{row['Model']} on {row['Platform']}: {row['Performance']:.2f} ms ± {row['StdError']:.2f} (SE), n={row['NumSamples']}"
        )

    # Generate the figures
    create_paper_figures(df, args.output_dir)
    print(f"Figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

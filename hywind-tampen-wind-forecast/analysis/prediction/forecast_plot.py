"""Plot XGBoost wind-speed forecasts against observations for one validation sample."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# =============================================================
# Paths
# Resolved relative to this file, so the script works from any
# working directory as long as the repo layout is intact:
#   <repo>/analysis/prediction/forecast_plot.py
#   <repo>/data/validation_sample_*.csv
# =============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV = DATA_DIR / "validation_sample_90_20241009_0522.csv"


# =============================================================
# Helpers
# =============================================================
def _fmt(value):
    """Format a value for a legend label, tolerating NaN."""
    return "n/a" if pd.isna(value) else f"{value:.2f}"


def _add_time_line(fig, t, color, dash, width):
    """Vertical line at time t, spanning the plot area.

    add_shape is used instead of add_vline because add_vline averages
    x0 and x1 whenever an annotation is attached, which fails on
    pandas Timestamps.
    """
    fig.add_shape(
        type="line",
        x0=t,
        x1=t,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            color=color,
            dash=dash,
            width=width
        )
    )


def plot_forecast_plotly(
    df,
    sample_id="90",
    station="HY09",
    tolerance=pd.Timedelta(minutes=5),
    show=True
):

    # =========================================================
    # Prepare data
    # =========================================================
    df = df.copy()

    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)

    # =========================================================
    # Identify forecast origin
    # =========================================================
    forecast_mask = df["Forecast"].notna()

    if not forecast_mask.any():
        raise ValueError("No non-NaN values found in 'Forecast'.")

    origin_time = df.loc[df.index[forecast_mask][0], "Time"]

    history = df[df["Time"] <= origin_time]
    future = df[df["Time"] >= origin_time]

    # =========================================================
    # Forecast horizons
    # =========================================================
    t30 = origin_time + pd.Timedelta(minutes=30)
    t60 = origin_time + pd.Timedelta(minutes=60)

    def nearest_row(t):
        """Row closest to t, or an error if nothing lies within tolerance."""
        diffs = (df["Time"] - t).abs()
        idx = diffs.idxmin()

        if diffs.loc[idx] > tolerance:
            raise ValueError(
                f"No observation within {tolerance} of {t:%Y-%m-%d %H:%M}. "
                f"Sample covers {df['Time'].min():%Y-%m-%d %H:%M} to "
                f"{df['Time'].max():%Y-%m-%d %H:%M}."
            )

        return df.loc[idx]

    row30 = nearest_row(t30)
    row60 = nearest_row(t60)

    # =========================================================
    # Validation metrics
    # Restricted to the 0-60 min horizon, which is what the
    # title claims. Without the t60 cut these would cover the
    # full length of the sample.
    # =========================================================
    eval_df = (
        future[future["Time"] <= t60]
        .dropna(subset=["Actual", "Forecast"])
    )

    if len(eval_df) > 0:
        errors = eval_df["Actual"] - eval_df["Forecast"]

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
    else:
        rmse = np.nan
        mae = np.nan

    # =========================================================
    # Create figure
    # =========================================================
    fig = go.Figure()

    # =========================================================
    # Historical observations
    # =========================================================
    fig.add_trace(
        go.Scatter(
            x=history["Time"],
            y=history["Actual"],
            mode="lines",
            line=dict(
                color="black",
                width=2.5
            ),
            name="Observed history",
            legendgroup="series"
        )
    )

    # =========================================================
    # Actual future observations
    # =========================================================
    fig.add_trace(
        go.Scatter(
            x=future["Time"],
            y=future["Actual"],
            mode="lines",
            line=dict(
                color="forestgreen",
                width=3
            ),
            name="Actual future",
            legendgroup="series"
        )
    )

    # =========================================================
    # XGBoost forecast
    # =========================================================
    fig.add_trace(
        go.Scatter(
            x=future["Time"],
            y=future["Forecast"],
            mode="lines",
            line=dict(
                color="royalblue",
                width=3,
                dash="dash"
            ),
            name="XGBoost forecast",
            legendgroup="series"
        )
    )

    # =========================================================
    # Forecast origin
    # Line and label are added separately; see _add_time_line.
    # =========================================================
    _add_time_line(
        fig,
        origin_time,
        color="gray",
        dash="dash",
        width=2
    )

    fig.add_annotation(
        x=origin_time,
        y=1,
        xref="x",
        yref="paper",
        yanchor="bottom",
        text="Forecast origin",
        showarrow=False,
        font=dict(size=12)
    )

    # =========================================================
    # +30 minute vertical line
    # =========================================================
    _add_time_line(
        fig,
        t30,
        color="lightsteelblue",
        dash="dot",
        width=2.5
    )

    # =========================================================
    # +60 minute vertical line
    # =========================================================
    _add_time_line(
        fig,
        t60,
        color="goldenrod",
        dash="dot",
        width=3
    )

    # =========================================================
    # +30 forecast marker
    # =========================================================
    fig.add_trace(
        go.Scatter(
            x=[t30],
            y=[row30["Forecast"]],
            mode="markers",
            marker=dict(
                color="royalblue",
                size=12,
                line=dict(
                    color="white",
                    width=1
                )
            ),
            name=f"Forecast +30 min: {_fmt(row30['Forecast'])}",
            legendgroup="markers"
        )
    )

    # =========================================================
    # +30 actual marker
    # Green X with dark outline
    # =========================================================
    fig.add_trace(
        go.Scatter(
            x=[t30],
            y=[row30["Actual"]],
            mode="markers",
            marker=dict(
                color="limegreen",
                size=13,
                symbol="x",
                line=dict(
                    color="darkgreen",
                    width=2
                )
            ),
            name=f"Actual +30 min: {_fmt(row30['Actual'])}",
            legendgroup="markers"
        )
    )

    # =========================================================
    # +60 forecast marker
    # =========================================================
    fig.add_trace(
        go.Scatter(
            x=[t60],
            y=[row60["Forecast"]],
            mode="markers",
            marker=dict(
                color="orange",
                size=12,
                line=dict(
                    color="white",
                    width=1
                )
            ),
            name=f"Forecast +60 min: {_fmt(row60['Forecast'])}",
            legendgroup="markers"
        )
    )

    # =========================================================
    # +60 actual marker
    # Red X with dark outline
    # =========================================================
    fig.add_trace(
        go.Scatter(
            x=[t60],
            y=[row60["Actual"]],
            mode="markers",
            marker=dict(
                color="red",
                size=13,
                symbol="x",
                line=dict(
                    color="darkred",
                    width=2
                )
            ),
            name=f"Actual +60 min: {_fmt(row60['Actual'])}",
            legendgroup="markers"
        )
    )

    # =========================================================
    # Metrics
    # =========================================================
    if np.isnan(rmse):
        metrics_text = "0-60 min RMSE: n/a | MAE: n/a"
    else:
        metrics_text = (
            f"0-60 min RMSE: {rmse:.3f} | "
            f"MAE: {mae:.3f} "
            f"(n={len(eval_df)})"
        )

    # =========================================================
    # Layout
    # =========================================================
    fig.update_layout(

        # -----------------------------------------------------
        # Title
        # -----------------------------------------------------
        title=dict(
            text=(
                f"<b>{station} Wind Forecast</b>"
                f"<br>"
                f"<span style='font-size:13px'>"
                f"Validation sample {sample_id} · "
                f"Origin: {origin_time:%Y-%m-%d %H:%M} · "
                f"{metrics_text}"
                f"</span>"
            ),
            x=0.5,
            xanchor="center",
            y=0.97
        ),

        # -----------------------------------------------------
        # X axis
        # -----------------------------------------------------
        xaxis=dict(
            title=dict(
                text="Time",
                font=dict(size=13)
            ),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.10)",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True
        ),

        # -----------------------------------------------------
        # Y axis
        # -----------------------------------------------------
        yaxis=dict(
            title=dict(
                text="Wind speed (m/s)",
                font=dict(size=13)
            ),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.10)",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True
        ),

        # -----------------------------------------------------
        # Publication-style legend
        # -----------------------------------------------------
        legend=dict(
            x=1.02,
            y=0.98,
            xanchor="left",
            yanchor="top",

            bgcolor="white",
            bordercolor="black",
            borderwidth=1,

            font=dict(
                size=13
            ),

            itemsizing="constant",

            tracegroupgap=8
        ),

        # -----------------------------------------------------
        # Margins
        # -----------------------------------------------------
        margin=dict(
            l=75,
            r=310,
            t=100,
            b=70
        ),

        # -----------------------------------------------------
        # General appearance
        # -----------------------------------------------------
        template="plotly_white",

        width=1250,
        height=600,

        hovermode="x unified",

        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    # =========================================================
    # Display
    # =========================================================
    if show:
        fig.show()

    return fig


# =============================================================
# Run plot
# =============================================================
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Validation CSV to plot (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--sample-id",
        default="90"
    )
    parser.add_argument(
        "--station",
        default="HY09"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"CSV not found: {args.csv}\n"
            f"Expected the data directory at: {DATA_DIR}"
        )

    result_data = pd.read_csv(args.csv)

    plot_forecast_plotly(
        result_data,
        sample_id=args.sample_id,
        station=args.station
    )
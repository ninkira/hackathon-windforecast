import pandas as pd
import numpy as np
import plotly.graph_objects as go


def plot_forecast_plotly(df, sample_id="90", station="HY09"):

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

    origin_idx = forecast_mask.idxmax()
    origin_time = df.loc[origin_idx, "Time"]

    history = df[df["Time"] <= origin_time]
    future = df[df["Time"] >= origin_time]

    # =========================================================
    # Forecast horizons
    # =========================================================
    t30 = origin_time + pd.Timedelta(minutes=30)
    t60 = origin_time + pd.Timedelta(minutes=60)

    def nearest_row(t):
        idx = (df["Time"] - t).abs().idxmin()
        return df.loc[idx]

    row30 = nearest_row(t30)
    row60 = nearest_row(t60)

    # =========================================================
    # Validation metrics
    # =========================================================
    eval_df = future.dropna(subset=["Actual", "Forecast"])

    if len(eval_df) > 0:
        rmse = np.sqrt(
            np.mean(
                (eval_df["Actual"] - eval_df["Forecast"]) ** 2
            )
        )

        mae = np.mean(
            np.abs(
                eval_df["Actual"] - eval_df["Forecast"]
            )
        )
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
    # =========================================================
    fig.add_vline(
        x=origin_time,
        line=dict(
            color="gray",
            dash="dash",
            width=2
        ),
        annotation_text="Forecast origin",
        annotation_position="top"
    )

    # =========================================================
    # +30 minute vertical line
    # =========================================================
    fig.add_vline(
        x=t30,
        line=dict(
            color="lightsteelblue",
            dash="dot",
            width=2.5
        )
    )

    # =========================================================
    # +60 minute vertical line
    # =========================================================
    fig.add_vline(
        x=t60,
        line=dict(
            color="goldenrod",
            dash="dot",
            width=3
        )
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
            name=f"Forecast +30 min: {row30['Forecast']:.2f}",
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
            name=f"Actual +30 min: {row30['Actual']:.2f}",
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
            name=f"Forecast +60 min: {row60['Forecast']:.2f}",
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
            name=f"Actual +60 min: {row60['Actual']:.2f}",
            legendgroup="markers"
        )
    )

    # =========================================================
    # Metrics
    # =========================================================
    if np.isnan(rmse):
        metrics_text = "RMSE: n/a | MAE: n/a"
    else:
        metrics_text = (
            f"60-min RMSE: {rmse:.3f} | "
            f"MAE: {mae:.3f}"
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

            # Larger legend font
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
    fig.show()

    return fig


# =============================================================
# Run plot
# =============================================================

if __name__ == "__main__":

    result_data = pd.read_csv("C:\\Users\\506895\\swdevelopment\\MS Modeler\\ms_modeler\\hackathon\\data_file\\validation_sample_90_20241009_0522.csv")
    plot_forecast_plotly(
        result_data,
        sample_id="90",
        station="HY09"
    )
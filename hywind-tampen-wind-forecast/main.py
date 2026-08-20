import pandas as pd, os
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

def local_extrema(z, size=5, thresh=None):
    zmax = maximum_filter(z, size=size)
    zmin = minimum_filter(z, size=size)
    is_max = (z == zmax)
    is_min = (z == zmin)
    if thresh is not None:
        is_max &= z > thresh          # drop noise plateaus
        is_min &= z < -thresh
    return np.argwhere(is_max), np.argwhere(is_min)


if __name__ == '__main__':
    data_dir = "./data"
    train = pd.read_parquet(os.path.join(data_dir, "windfeels_train.parquet"))
    test  = pd.read_parquet(os.path.join(data_dir, "windfeels_test.parquet"))
    print(train.shape, test.shape)
    print(train.dtypes)
    print(train.head(10))
    print("\ncols only in train:", set(train.columns) - set(test.columns))
    print("missing frac:\n", train.isna().mean().sort_values(ascending=False).head(10))

    x = [120, 340, 890]  # pixel x
    y = [80, 410, 220]  # pixel y
    val = [0.91, 0.87, 0.95]  # for labels/hover
    max_rc, min_rc = local_extrema(val, size=7, thresh=np.percentile(val, 90))
    img = Image.open("./images/map.png")
    w, h = img.size

    heat = np.random.rand(50, 50)  # replace with your real grid

    x = [120, 340, 890]
    y = [80, 410, 220]
    val = [0.91, 0.87, 0.95]

    fig = go.Figure()

    fig.add_layout_image(
        source=img, xref="x", yref="y",
        x=0, y=0, sizex=w, sizey=h,
        sizing="stretch", layer="below",
    )

    fig.add_trace(go.Heatmap(
        z=heat,
        x=np.linspace(0, w, heat.shape[1]),
        y=np.linspace(0, h, heat.shape[0]),
        opacity=0.5, colorscale="Inferno", zsmooth="best",
    ))

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text",
        marker=dict(symbol="triangle-up", size=12,
                    color="rgba(0,0,0,0)", line=dict(color="#e34948", width=2)),
        text=[f"{v:.2f}" for v in val],
        textposition="top center",
        textfont=dict(color="#e34948", size=11),
        name="points",
    ))

    fig.update_xaxes(range=[0, w], visible=False)
    fig.update_yaxes(range=[h, 0], visible=False, scaleanchor="x")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.show()
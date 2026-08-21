import pandas as pd, os
import plotly
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from cv2.gapi.ot import NEW
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

    print(test.shape, test.shape)
    print(test.dtypes)
    print(test.head(10))
    print("\ncols only in train:", set(train.columns) - set(test.columns))
    print("missing frac:\n", train.isna().mean().sort_values(ascending=False).head(10))




    x = [120, 340, 890]  # pixel x
    y = [80, 410, 220]  # pixel y
    val = [0.91, 0.87, 0.95]  # for labels/hover
    max_rc, min_rc = local_extrema(val, size=7, thresh=np.percentile(val, 90))
    img = Image.open("./images/map.png")
    w, h = img.size

        # ---------------------------------------------------------------
        # Dummy data: three clusters of turbines off the Norwegian coast
        # ---------------------------------------------------------------
    rng = np.random.default_rng(7)

    clusters = [
        (61.33, 2.28, 25),  # lat, lon, how many turbines
        (61.05, 2.55, 18),
        (61.55, 1.95, 12),
    ]

    frames = []
    for i, (lat0, lon0, n) in enumerate(clusters):
        frames.append(pd.DataFrame({
            "lat": rng.normal(lat0, 0.09, n),
            "lon": rng.normal(lon0, 0.16, n),
            "wind": rng.normal(14, 2.5, n).clip(5, 25),
            "site": [f"C{i}-T{j:02d}" for j in range(n)],
        }))

    df = pd.concat(frames, ignore_index=True)

    NEW = tuple(int(p) for p in plotly.__version__.split(".")[:2]) >= (5, 24)
    Density = go.Densitymap if NEW else go.Densitymapbox
    Scatter = go.Scattermap if NEW else go.Scattermapbox
    MAP_KEY = "map" if NEW else "mapbox"

    # ---------------------------------------------------------------
    # Colourscale: transparent at the bottom so the basemap shows
    # through where there is no data. Without the rgba(0,0,0,0) stop
    # you get an opaque wash over the whole viewport.
    # ---------------------------------------------------------------
    SCALE = [
        [0.00, "rgba(0,0,0,0)"],
        [0.25, "#3ecf4a"],
        [0.55, "#ffd400"],
        [0.80, "#ff7a1a"],
        [1.00, "#e02020"],
    ]

    fig = go.Figure()

    fig.add_trace(Density(
        lat=df.lat,
        lon=df.lon,
        z=df.wind,
        radius=45,  # screen pixels, not km - changes with zoom
        opacity=0.65,
        colorscale=SCALE,
        colorbar=dict(title="m/s"),
        hoverinfo="skip",
    ))

    fig.add_trace(Scatter(
        lat=df.lat,
        lon=df.lon,
        mode="markers",
        marker=dict(size=df.wind * 0.9, color="#c0392b", opacity=0.9),
        text=[f"{s}: {w:.1f} m/s" for s, w in zip(df.site, df.wind)],
        hovertemplate="%{text}<extra></extra>",
        name="turbines",
    ))

    # carto-positron / open-street-map / carto-darkmatter need no token.
    # "satellite" only works on plotly >= 5.24.
    fig.update_layout(**{
        MAP_KEY: dict(
            style="carto-positron",
            center=dict(lat=61.3, lon=2.3),
            zoom=7.5,
        ),
        "margin": dict(l=0, r=0, t=0, b=0),
        "height": 620,
    })


    print(f"plotly {plotly.__version__} - using {'map' if NEW else 'mapbox'} trace names")
    fig.write_html("wind_density_map.html", auto_open=False)
    print("wrote wind_density_map.html")
    fig.show()
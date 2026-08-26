"""
Wind speed heatmap + direction arrows from station u/v data.

Expected input schema (wide format):
    timestamp   | episode_id | <station>_u | <station>_v | ...
    2024-01-01  | EP0001     |  3.4        | -1.2        |

Pipeline:
    parquet -> long format -> interpolate stations onto a grid
            -> speed = hypot(u, v) -> heatmap + quiver -> animation

The only thing you MUST fill in is STATIONS: the position of each
station. Without positions there is no spatial field to interpolate.

Run:  python wind_field.py
"""

import os
import re
import sys
import base64
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter, minimum_filter

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
# Everything resolves relative to THIS file, not to the working directory.
# PyCharm often runs with the project root as the working directory, which is
# why "./data/..." works from the terminal but not from the IDE.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))      # so `import forecast_plot` finds the
                                       # file sitting next to this one
DATA_DIR = HERE / "data"
IMAGE_PATH = HERE / "images" / "map.png"   # the 524x551 field map

TIME_COL = "timestamp"      # auto-detected if this name is not in the file
EPISODE_COL = "episode_id"  # same

EPISODE_CUTOFF = "EP0344"   # keep episodes up to and including this one
FRAME_BY = "timestamp"      # "timestamp" = one frame per minute WITHIN one
                            #   episode - continuous, no jump cuts
                            # "episode" = one frame per episode. Episodes are
                            #   separate windows, so this flickers by design
EPISODE = "last"            # which episode to animate when FRAME_BY is
                            # "timestamp": "last", "first", or e.g. "EP0344"
MAX_FRAMES = 60             # hard cap - every frame is stored in the HTML
EPISODE_GAP = "10min"       # only used if the file has no episode column

GRID = (60, 80)             # field rows, cols

# One HTML file is written per entry. Options:
#   "speed" = hypot(u, v)   - never negative, scale starts at 0
#   "u"     = east-west     - signed, diverging scale (- is westward)
#   "v"     = north-south   - signed, diverging scale (- is southward)
#   "abs_u" / "abs_v"       - strength only, sign dropped, scale starts at 0
COMPONENTS = ["speed"]      # one HTML file per entry, e.g. ["speed", "u", "v"]

FIELD_STYLE = "heatmap"     # "contour" = filled bands like matplotlib contourf
                            # "heatmap" = smooth continuous shading
CONTOUR_STEP = 1.0          # m/s per band, matches np.arange(0, 14, 1)
# Named entries below, or any Plotly scale name ("Jet", "Turbo", "Viridis",
# "GnBu", "YlGnBu", "Blues"), or your own [[pos, colour], ...] list.
PALETTES = {
    # low = pale blue, high = deep green
    "blue_green": [
        [0.00, "#deebf7"], [0.18, "#9ecae1"], [0.34, "#4292c6"],
        [0.50, "#2171b5"], [0.62, "#c7e9b4"], [0.76, "#78c679"],
        [0.88, "#41880f"], [1.00, "#1a4a10"],
    ],
    # low = deep green, high = deep blue (the reverse reading)
    "green_blue": [
        [0.00, "#1a4a10"], [0.14, "#41880f"], [0.28, "#78c679"],
        [0.42, "#c7e9b4"], [0.56, "#2171b5"], [0.72, "#4292c6"],
        [0.86, "#9ecae1"], [1.00, "#deebf7"],
    ],
    # calm -> strong, colour-blind safe, no Jet artefacts
    "cool_warm": [
        [0.00, "#f7fbff"], [0.25, "#9ecae1"], [0.50, "#4eb3d3"],
        [0.75, "#f0a202"], [1.00, "#b02e0c"],
    ],
}

COLORSCALE = "blue_green"   # used for "speed" / "abs_u" / "abs_v"
DIVERGING = "RdBu"          # used for "u"/"v" - signed data needs a diverging scale


def palette(name):
    """Accept a PALETTES key, a Plotly scale name, or a literal list."""
    return PALETTES.get(name, name) if isinstance(name, str) else name
FIELD_OPACITY = 0.65        # lower = more of the map image shows through
CLIP_PCT = 98               # colour limits from this percentile, not the extremes.
                            # 100 = true min/max (one gust flattens everything else)
                            # 90-98 = weak wind stays visible, strongest gusts clip

ARROW_EVERY = 6             # draw one arrow per N grid cells (higher = fewer)
ARROW_SCALE = 1.75          # arrow length (lower = shorter)
ARROW_HEAD = 0.35           # head size as a fraction of shaft
ARROW_WIDTH = 0.7           # line thickness

DISPLAY_WIDTH = 800
CONTROL_H = 90              # px reserved BELOW the map for slider + buttons
TITLE_H = 0                 # px above the map for a title (0 = none)
FRAME_MS = 500              # ms per frame while playing
TRANSITION_MS = 0           # tweening between frames. Keep 0: Plotly cannot
                            # tween a contour/heatmap, so a non-zero value
                            # blanks the plot area while it redraws
SUBFRAMES = 2               # interpolated frames inserted BETWEEN measurements
SMOOTH_WINDOW = 3           # rolling mean over N timesteps per station (1 = off)
LOOP = True                 # restart automatically when the last frame ends
AUTOPLAY = False            # False = opens paused; the viewer presses Play.
                            # Safer default: a looping full-area animation can
                            # be uncomfortable for photosensitive viewers.
SHOW = True                 # also open each figure in the browser
N_EXTREMA = 0               # local max/min triangles per frame (0 = off)
SHOW_STATIONS = True        # the labelled rings at each station
STAMP = True                # per-frame timestamp drawn ON the map
STAMP_POS = (0.02, 0.05)    # fraction of width/height, from the top left

PAGE_HEADLINE = "Wind Forecast Portal"
PAGE_SUBTITLE = "Hywind Tampen and the surrounding Tampen platforms"

# Second graphic: the XGBoost forecast plot. Set the CSV to None to keep
# the placeholder. The module must expose plot_forecast_plotly(df, ...).
FORECAST_CSV = HERE / "data" / "validation_sample_90_20241009_0522.csv"
FORECAST_MODULE = "forecast_plot"   # the .py file next to this one
FORECAST_FUNC = None                # None = auto-detect the plotting function
FORECAST_SAMPLE = "90"
FORECAST_STATION = "HY09"

# KPI tiles across the top. tone: "" | "good" | "warn" | "bad"
KPIS = [
    dict(label="Next period", value="1,284", note="80% PI: 1,090-1,478"),
    dict(label="vs. last period", value="+6.2%", note="1,209 -> 1,284", tone="good"),
    dict(label="Backtest MAPE", value="7.4%", note="naive baseline: 12.9%"),
    dict(label="PI coverage", value="71%", note="nominal 80% - too narrow", tone="warn"),
]

# Station positions in IMAGE PIXELS (origin top-left, y downward).
# Keys are matched case- and separator-insensitively, so "Statfjord_A",
# "statfjord a" and "STATFJORDA" all refer to the same station.
# Replace these with the real layout.
STATIONS = {
    # Platforms - pixel centres of the orange rings on map.png (524x551).
    # Which ring is A vs B is a guess (north to south); check against the
    # real layout and swap if needed.
    "Statfjord_A": (105, 290),
    "Statfjord_B": (93, 299),
    "Gullfaks_C":  (283, 336),
    "Snorre_A":    (224, 115),
    "Snorre_B":    (254, 42),
    "Visund":      (367, 190),

    # Hywind Tampen - 11 turbines spread along the green field outline.
    "HYT-HY01": (274, 192),
    "HYT-HY02": (265, 201),
    "HYT-HY03": (278, 204),
    "HYT-HY04": (269, 213),
    "HYT-HY05": (281, 217),
    "HYT-HY06": (272, 226),
    "HYT-HY07": (284, 229),
    "HYT-HY08": (275, 238),
    "HYT-HY09": (287, 242),
    "HYT-HY10": (278, 251),
    "HYT-HY11": (290, 254),
}


def norm(s) -> str:
    """statfjord_a / Statfjord A / STATFJORDA -> statfjorda"""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


STATION_POS = {norm(k): v for k, v in STATIONS.items()}


# ---------------------------------------------------------------
# 1. Reshape: wide station columns -> long format
# ---------------------------------------------------------------
def find_station_columns(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Pair up columns ending in u / v by their shared prefix.

    Statfjord_A_U + Statfjord_A_V  ->  station 'Statfjord_A'
    """
    parts: dict[str, dict[str, str]] = {}
    for col in df.columns:
        m = re.match(r"^(?P<st>.+?)[ _\-]*(?P<c>[uv])$", str(col).strip(), re.IGNORECASE)
        if m:
            parts.setdefault(m.group("st"), {})[m.group("c").lower()] = col
    pairs = {st: (d["u"], d["v"]) for st, d in parts.items() if "u" in d and "v" in d}
    if not pairs:
        raise ValueError(
            "No u/v column pairs found. Columns are:\n  " + "\n  ".join(map(str, df.columns))
        )
    return pairs


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (timestamp, episode, station) with u, v, speed, direction."""
    pairs = find_station_columns(df)
    frames = []
    for station, (ucol, vcol) in pairs.items():
        frames.append(pd.DataFrame({
            TIME_COL: df[TIME_COL],
            EPISODE_COL: df[EPISODE_COL],
            "station": station,
            "u": df[ucol].astype("float32"),
            "v": df[vcol].astype("float32"),
        }))
    long = pd.concat(frames, ignore_index=True)

    long["speed"] = np.hypot(long.u, long.v)
    # meteorological convention: direction the wind blows FROM, in degrees
    long["dir_from"] = (270 - np.degrees(np.arctan2(long.v, long.u))) % 360
    return long


# ---------------------------------------------------------------
# 2. Interpolate scattered stations onto a regular grid
# ---------------------------------------------------------------
def interpolate_field(sub: pd.DataFrame, gx, gy, stations: dict):
    """
    sub: rows for ONE timestamp. Returns (speed, u, v) grids.

    Linear inside the station hull, nearest-neighbour outside, because
    linear returns NaN beyond the convex hull of the points.
    """
    known = sub[sub.station.map(lambda s: norm(s) in stations)]
    # a frame may hold many rows per station (a whole episode); average them
    known = known.groupby("station", as_index=False)[["u", "v"]].mean()
    if len(known) < 3:
        raise ValueError(
            f"only {len(known)} of the stations have coordinates - need >= 3.\n"
            f"  stations in the data: {sorted(sub.station.unique())}\n"
            f"  keys in STATIONS:     {sorted(STATIONS)}\n"
            f"  -> add the missing stations to STATIONS with their pixel positions."
        )

    pts = np.array([stations[norm(s)] for s in known.station])
    X, Y = np.meshgrid(gx, gy)

    def grid(vals):
        lin = griddata(pts, vals, (X, Y), method="linear")
        near = griddata(pts, vals, (X, Y), method="nearest")
        return np.where(np.isnan(lin), near, lin)

    gu = grid(known.u.to_numpy())
    gv = grid(known.v.to_numpy())

    # z is recomputed per component at figure-build time; keep speed here
    return np.hypot(gu, gv), gu, gv


# ---------------------------------------------------------------
# 3. Load
# ---------------------------------------------------------------
def promote_index(df: pd.DataFrame) -> pd.DataFrame:
    """Move anything living in the index into real columns.

    Covers a plain DatetimeIndex and a MultiIndex such as
    (timestamp, episode_id) - a common way this data gets saved.
    """
    if isinstance(df.index, pd.MultiIndex):
        print(f"index levels promoted to columns: {list(df.index.names)}")
        return df.reset_index()
    if not isinstance(df.index, pd.RangeIndex):
        name = df.index.name or "timestamp"
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: name})
        print(f"promoted the index to column '{name}'")
    return df


def synthesize_episodes(df: pd.DataFrame, time_col: str, gap: str = "10min") -> pd.DataFrame:
    """
    Build an episode_id when the file has none.

    Consecutive rows closer together than `gap` belong to the same window;
    a larger jump starts a new episode. Check the printed episode count
    against what you expect - if it is wrong, adjust `gap`.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    t = pd.to_datetime(df[time_col])
    new_window = t.diff() > pd.Timedelta(gap)
    idx = new_window.cumsum() + 1
    df["episode_id"] = "EP" + idx.astype(int).astype(str).str.zfill(4)
    print(f"no episode column - derived {idx.max()} episodes from gaps > {gap}")
    return df


def resolve_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Find the time and episode columns whatever they are actually called."""
    global TIME_COL, EPISODE_COL

    if TIME_COL not in df.columns:
        cand = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if not cand:
            cand = [c for c in df.columns if re.search(r"time|date|ts$", c, re.I)]
        if not cand:
            raise KeyError(
                f"no time column found. Columns are:\n  " + "\n  ".join(map(str, df.columns))
            )
        TIME_COL = cand[0]

    if EPISODE_COL not in df.columns:
        cand = [c for c in df.columns if re.search(r"episode|ep_?id|window|event", c, re.I)]
        if not cand:  # fall back: a column whose values look like EP0001
            cand = [c for c in df.columns
                    if df[c].dtype == object
                    and df[c].astype(str).str.match(r"^EP\d+$").any()]
        EPISODE_COL = cand[0] if cand else None

    print(f"using time column '{TIME_COL}', episode column {EPISODE_COL!r}")
    return TIME_COL, EPISODE_COL


def episode_num(s) -> int:
    """EP0344 -> 344. Used so the cutoff sorts numerically, not as text."""
    digits = re.sub(r"\D", "", str(s))
    return int(digits) if digits else -1


for label, path in [("data dir", DATA_DIR), ("map image", IMAGE_PATH),
                    ("forecast csv", FORECAST_CSV)]:
    print(f"{label:13s} {path}  {'ok' if Path(path).exists() else 'MISSING'}")

train = pd.read_parquet(os.path.join(DATA_DIR, "windfeels_train.parquet"))
train = train.dropna()
train = promote_index(train)
print("columns:", train.columns.tolist())
resolve_columns(train)

if EPISODE_COL is None:
    train = synthesize_episodes(train, TIME_COL, gap=EPISODE_GAP)
    EPISODE_COL = "episode_id"

long = to_long(train)

# Drop only the (station, timestamp) readings that are actually missing.
# train.dropna() would delete the whole row - every station at that minute -
# because one station had a gap. Here the other 16 stations survive.
before = len(long)
gaps = long[long[["u", "v"]].isna().any(axis=1)].station.value_counts()
long = long.dropna(subset=["u", "v"])
if len(long) < before:
    print(f"dropped {before - len(long)} missing station readings "
          f"({100 * (before - len(long)) / before:.1f}%)")
    for st, n in gaps.head(8).items():
        print(f"    {st}: {n} missing")
if long.empty:
    raise ValueError("every reading is NaN - check the file")

if SMOOTH_WINDOW > 1:
    long = long.sort_values([TIME_COL])
    long[["u", "v"]] = (
        long.groupby("station")[["u", "v"]]
            .transform(lambda c: c.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean())
    )
    long["speed"] = np.hypot(long.u, long.v)
    print(f"smoothed u/v with a centred rolling mean of {SMOOTH_WINDOW} steps")

print("stations found:", sorted(long.station.unique()))

all_eps = sorted(long[EPISODE_COL].unique(), key=episode_num)
print(f"episodes in file: {len(all_eps)} ({all_eps[0]} .. {all_eps[-1]})")

missing = {s for s in long.station.unique() if norm(s) not in STATION_POS}
if missing:
    print(f"WARNING: no coordinates for {sorted(missing)} - they are ignored")

# --- the cutoff: keep everything up to AND INCLUDING EPISODE_CUTOFF ---
cut = episode_num(EPISODE_CUTOFF)
long = long[long[EPISODE_COL].map(episode_num) <= cut]
if long.empty:
    raise ValueError(f"no rows at or before {EPISODE_CUTOFF}")

kept = sorted(long[EPISODE_COL].unique(), key=episode_num)
print(f"kept {len(kept)} episodes ({kept[0]} .. {kept[-1]}), {len(long)} rows")

# --- group into animation frames ---
if FRAME_BY == "episode":
    keys = kept[:MAX_FRAMES]
    groups = [long[long[EPISODE_COL] == k] for k in keys]
    labels = [str(k) for k in keys]
else:
    # Stay inside ONE episode. Stepping across episode boundaries jumps
    # between unrelated weather windows and looks like a strobe.
    pick = {"last": kept[-1], "first": kept[0]}.get(EPISODE, EPISODE)
    if pick not in set(kept):
        raise ValueError(f"episode {pick!r} not in the kept range "
                         f"({kept[0]} .. {kept[-1]})")
    within = long[long[EPISODE_COL] == pick]
    keys = sorted(within[TIME_COL].unique())[:MAX_FRAMES]
    groups = [within[within[TIME_COL] == k] for k in keys]
    labels = [str(pd.Timestamp(k))[11:16] for k in keys]
    print(f"animating within {pick}: {len(keys)} timesteps "
          f"({labels[0]} to {labels[-1]})")

if len(kept if FRAME_BY == "episode" else keys) > MAX_FRAMES:
    print(f"NOTE: capped at {MAX_FRAMES} frames - raise MAX_FRAMES to see more")
print(f"{len(groups)} frames, one per {FRAME_BY}")

# ---------------------------------------------------------------
# 4. Background image
# ---------------------------------------------------------------
if IMAGE_PATH.exists():
    img = Image.open(IMAGE_PATH)
else:
    print(f"no image at {IMAGE_PATH} - using a grey placeholder")
    img = Image.new("RGB", (1200, 800), (228, 228, 222))

W, H = img.size
DISPLAY_HEIGHT = round(DISPLAY_WIDTH * H / W)

gx = np.linspace(0, W, GRID[1])
gy = np.linspace(0, H, GRID[0])
GX, GY = np.meshgrid(gx, gy)

# ---------------------------------------------------------------
# 5. Precompute every frame
# ---------------------------------------------------------------
fields, kept_labels, skipped = [], [], 0
for g, lab in zip(groups, labels):
    try:
        fields.append(interpolate_field(g, gx, gy, STATION_POS))
        kept_labels.append(lab)
    except ValueError as e:
        skipped += 1
        if skipped == 1:
            print(f"skipping frames with too few stations: {e.args[0].splitlines()[0]}")
labels = kept_labels
if skipped:
    print(f"skipped {skipped} frames that had fewer than 3 reporting stations")
if not fields:
    raise ValueError("no frame had 3+ stations with coordinates")


def component(gu, gv, comp=None):
    """Turn the two vector grids into the scalar field being coloured."""
    comp = comp or COMPONENTS[0]
    if comp == "u":
        return gu                    # signed: + eastward, - westward
    if comp == "v":
        return gv                    # signed: + northward, - southward
    if comp == "abs_u":
        return np.abs(gu)            # east-west strength, sign dropped
    if comp == "abs_v":
        return np.abs(gv)
    return np.hypot(gu, gv)          # speed, never negative


if SUBFRAMES > 1 and len(fields) > 1:
    # Linear blend between consecutive measured fields. Interpolating the
    # COMPONENTS (not the speed) keeps direction and magnitude consistent:
    # a vector turning from east to north passes through north-east, it does
    # not shrink to zero and grow back.
    dense, dense_labels = [], []
    for i in range(len(fields) - 1):
        (_, u0, v0), (_, u1, v1) = fields[i], fields[i + 1]
        for k in range(SUBFRAMES):
            a = k / SUBFRAMES
            gu = (1 - a) * u0 + a * u1
            gv = (1 - a) * v0 + a * v1
            dense.append((np.hypot(gu, gv), gu, gv))
            dense_labels.append(labels[i] if k == 0 else f"{labels[i]} +{k}/{SUBFRAMES}")
    dense.append(fields[-1])
    dense_labels.append(labels[-1])
    fields, labels = dense, dense_labels
    print(f"upsampled to {len(fields)} frames ({SUBFRAMES}x between measurements)")


def markers() -> go.Scatter:
    """Static ring for every station that has coordinates."""
    names = [s for s in sorted(long.station.unique()) if norm(s) in STATION_POS]
    return go.Scatter(
        x=[STATION_POS[norm(s)][0] for s in names],
        y=[STATION_POS[norm(s)][1] for s in names],
        mode="markers+text",
        marker=dict(size=7, color="rgba(0,0,0,0)",
                    line=dict(color="#111111", width=2)),
        text=names, textposition="bottom center",
        textfont=dict(size=10, color="#111111"),
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    )


LABELS = {
    "speed": "Wind speed",
    "u": "East-west wind (U)",
    "v": "North-south wind (V)",
    "abs_u": "East-west strength |U|",
    "abs_v": "North-south strength |V|",
}

SEQUENTIAL = ("speed", "abs_u", "abs_v")   # start the colour scale at zero
DIVERGENT = ("u", "v")                     # signed, zero in the middle


def build_figure(comp: str) -> go.Figure:
    zs = [component(gu, gv, comp) for _, gu, gv in fields]

    flat = np.concatenate([z.ravel() for z in zs])
    if comp in DIVERGENT:
        # Signed field. Symmetric limits keep zero at the midpoint of the
        # colourscale, so + and - of equal size read as equally strong.
        # The percentile (not the max) sets the limit, otherwise a single
        # gust stretches the scale and every weak wind washes out to pale.
        lim = float(np.percentile(np.abs(flat), CLIP_PCT))
        zmin, zmax = -lim, lim
    else:
        zmin = 0.0                          # magnitude: anchor the scale at zero
        zmax = float(np.percentile(flat, CLIP_PCT))

    step = CONTOUR_STEP
    if (zmax - zmin) / step > 24:           # too many bands to read
        step = (zmax - zmin) / 16
    elif (zmax - zmin) / step < 5:          # too few: weak wind lands in one band
        step = (zmax - zmin) / 8

    def field(i):
        common = dict(
            z=zs[i], x=gx, y=gy, zmin=zmin, zmax=zmax,
            colorscale=palette(DIVERGING if comp in DIVERGENT else COLORSCALE),
            opacity=FIELD_OPACITY,
            colorbar=dict(title="m/s", thickness=14, len=0.85),
            hovertemplate="%{z:.1f} m/s<extra></extra>",
        )
        if FIELD_STYLE == "contour":
            return go.Contour(
                **common,
                contours=dict(start=zmin, end=zmax,
                              size=step, coloring="fill", showlines=False),
                line=dict(width=0),
            )
        return go.Heatmap(**common, zsmooth="best")

    def arrows(i):
        _, gu, gv = fields[i]
        s_ = ARROW_EVERY
        q = ff.create_quiver(
            GX[::s_, ::s_], GY[::s_, ::s_],
            gu[::s_, ::s_],
            -gv[::s_, ::s_],          # image y grows downward, north is up
            scale=ARROW_SCALE, arrow_scale=ARROW_HEAD,
            line=dict(width=ARROW_WIDTH, color="#1a1a1a"),
        )
        tr = q.data[0]
        tr.hoverinfo, tr.showlegend = "skip", False
        return tr

    def extrema(i):
        z = zs[i]
        if N_EXTREMA < 1:
            return go.Scatter(x=[], y=[], mode="markers", showlegend=False,
                              hoverinfo="skip")
        rank = np.abs(z) if comp in DIVERGENT else z
        hi = sorted(np.argwhere(rank == maximum_filter(rank, size=11)),
                    key=lambda rc: -rank[rc[0], rc[1]])[:N_EXTREMA]
        lo = sorted(np.argwhere(z == minimum_filter(z, size=11)),
                    key=lambda rc: z[rc[0], rc[1]])[:N_EXTREMA]
        pts = [(rc, "max") for rc in hi] + [(rc, "min") for rc in lo]
        return go.Scatter(
            x=[gx[c] for (r, c), _ in pts],
            y=[gy[r] for (r, c), _ in pts],
            mode="markers+text",
            marker=dict(
                symbol=["triangle-up" if k == "max" else "triangle-down" for _, k in pts],
                size=11, color="rgba(0,0,0,0)", line=dict(color="#111111", width=2)),
            text=[f"{z[r, c]:.1f}" for (r, c), _ in pts],
            textposition="top center", textfont=dict(size=10, color="#111111"),
            hovertemplate="%{text} m/s<extra></extra>", showlegend=False,
        )

    def stamp(i):
        """Per-frame timestamp as a DATA trace.

        A frame that changes layout (title, annotations) forces a Plotly
        relayout: the plot area is torn down, the axes are rebuilt and the
        background image is decoded again - a blank canvas for one frame.
        A text trace is plain data, so it updates with no repaint at all.
        """
        return go.Scatter(
            x=[W * STAMP_POS[0]], y=[H * STAMP_POS[1]],
            mode="text", text=[labels[i]],
            textposition="middle right",
            textfont=dict(size=15, color="#111111"),
            hoverinfo="skip", showlegend=False,
        )

    def title_for(i=0):
        span = f"{labels[0]} - {labels[-1]}" if len(labels) > 1 else labels[0]
        return f"{LABELS[comp]} and direction   {span}"

    base = [field(0), arrows(0), extrema(0), stamp(0)]
    if SHOW_STATIONS:
        base.append(markers())
    fig = go.Figure(data=base)
    fig.add_layout_image(source=img, xref="x", yref="y", x=0, y=0,
                         sizex=W, sizey=H, sizing="stretch", layer="below")
    # No layout= on the frames. A per-frame layout triggers a Plotly relayout,
    # which tears down and rebuilds the whole plot area - including re-decoding
    # the background image - and that blank repaint IS the white strobe.
    # The timestamp lives in the slider readout instead, which costs no redraw.
    fig.frames = [
        go.Frame(data=[field(i), arrows(i), extrema(i), stamp(i)],
                 traces=[0, 1, 2, 3], name=str(i))
        for i in range(len(fields))
    ]

    fig.update_xaxes(range=[0, W], visible=False, constrain="domain")
    fig.update_yaxes(range=[H, 0], visible=False, scaleanchor="x", constrain="domain")
    fig.update_layout(
        width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT + CONTROL_H + TITLE_H,
        margin=dict(l=0, r=0, t=TITLE_H, b=CONTROL_H),
        paper_bgcolor="rgba(0,0,0,0)",   # no white canvas to flash through
        plot_bgcolor="rgba(0,0,0,0)",
        transition=dict(duration=TRANSITION_MS),
        title=(dict(text=title_for(0), x=0.01, xanchor="left", y=0.99,
                    yanchor="top", font=dict(size=15)) if TITLE_H else None),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.0, y=-0.02, xanchor="left", yanchor="top",
            pad=dict(t=12, r=8), bgcolor="rgba(255,255,255,0.9)",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=FRAME_MS, redraw=True),
                                      transition=dict(duration=TRANSITION_MS),
                                      fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ])],
        sliders=[dict(
            active=0, x=0.16, y=-0.02, len=0.84, xanchor="left", yanchor="top",
            pad=dict(t=18, b=8), bgcolor="rgba(0,0,0,0.12)", borderwidth=0,
            tickcolor="rgba(0,0,0,0.35)", font=dict(size=10),
            currentvalue=dict(prefix="  ", font=dict(size=12)),
            steps=[dict(method="animate", label=lab,
                        args=[[str(i)], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True))])
                   for i, lab in enumerate(labels)],
        )],
    )
    print(f"  {comp}: scale {zmin:.1f} to {zmax:.1f} m/s, band {step:.2f}")
    if SHOW:
        fig.show()
    return fig


LOOP_JS = """
var gd = document.getElementById('{plot_id}');
var opts = {frame: {duration: %d, redraw: true},
            transition: {duration: %d},
            mode: 'immediate'};
function cycle() { Plotly.animate(gd, null, opts).then(function () {
    if (%s) { Plotly.animate(gd, ['0'], {frame: {duration: 0, redraw: false},
                                         transition: {duration: 0},
                                         mode: 'immediate'}).then(cycle); }
}); }
if (%s) { cycle(); }
""" % (FRAME_MS, TRANSITION_MS, "true" if LOOP else "false", "true" if AUTOPLAY else "false")



PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{headline}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0 24px 56px; background: #131313; color: #ededea;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  header {{ max-width: 1100px; margin: 0 auto; padding: 32px 0 8px; }}
  h1 {{ margin: 0; font-size: 28px; font-weight: 600; letter-spacing: -0.01em; }}
  .sub {{ margin: 6px 0 0; font-size: 15px; color: #9a9891; }}
  main {{ max-width: 1100px; margin: 0 auto; }}

  .kpis {{
    display: grid; gap: 16px; margin: 28px 0 8px;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  }}
  .kpi-label {{ font-size: 15px; color: #9a9891; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 30px; font-weight: 600; letter-spacing: -0.02em; }}
  .kpi-note {{ font-size: 14px; color: #9a9891; margin-top: 6px; line-height: 1.4; }}
  .good {{ color: #4ec26f; }} .warn {{ color: #e0a03a; }} .bad {{ color: #e05a4c; }}

  .tabs {{
    display: flex; gap: 4px; margin: 28px 0 0;
    border-bottom: 1px solid #2c2c2a;
  }}
  .tab {{
    appearance: none; border: 0; background: none; cursor: pointer;
    font: inherit; font-size: 15px; color: #9a9891;
    padding: 10px 16px; border-bottom: 2px solid transparent; margin-bottom: -1px;
  }}
  .tab:hover {{ color: #ededea; }}
  .tab.active {{ color: #ededea; border-bottom-color: #6f9ee8; }}

  .panel {{ display: none; padding: 20px 0 0; }}
  .panel.active {{ display: block; }}

  .placeholder {{
    border: 1px dashed #3a3a37; border-radius: 12px; padding: 64px 24px;
    text-align: center; color: #7c7a74; font-size: 15px; line-height: 1.7;
  }}
  .placeholder code {{
    background: #1e1e1c; padding: 2px 6px; border-radius: 4px; font-size: 13px;
  }}
  .plotwrap {{ background: #ffffff; border-radius: 12px; padding: 12px; overflow-x: auto; }}
  .legend {{
    display: flex; flex-wrap: wrap; gap: 20px;
    margin-top: 14px; font-size: 13px; color: #9a9891;
  }}
  footer {{
    max-width: 1100px; margin: 32px auto 0; font-size: 13px;
    color: #7c7a74; line-height: 1.6;
  }}
</style>
</head>
<body>
<header>
  <h1>{headline}</h1>
  <p class="sub">{subtitle}</p>
</header>
<main>
  <div class="kpis">{kpis}</div>

  <div class="tabs" role="tablist">
    <button class="tab active" role="tab" data-panel="p-forecast">Forecast</button>
    <button class="tab" role="tab" data-panel="p-field">Wind field</button>
  </div>

  <section class="panel active" id="p-forecast" role="tabpanel">
    {forecast}
  </section>

  <section class="panel" id="p-field" role="tabpanel">
    <div class="plotwrap">{plot}</div>
    <div class="legend">
      <span>Colour: wind speed in m/s</span>
      <span>Arrows: wind direction</span>
      <span>Rings: measurement stations</span>
    </div>
  </section>
</main>
<footer>
  Values between stations are interpolated, not measured. Outside the area
  covered by the stations the nearest station's value is carried across, so the
  edges of the map show the station layout rather than the wind.
</footer>
<script>
document.querySelectorAll('.tab').forEach(function (btn) {{
  btn.addEventListener('click', function () {{
    document.querySelectorAll('.tab').forEach(function (b) {{ b.classList.remove('active'); }});
    document.querySelectorAll('.panel').forEach(function (p) {{ p.classList.remove('active'); }});
    btn.classList.add('active');
    var panel = document.getElementById(btn.dataset.panel);
    panel.classList.add('active');
    // A Plotly chart laid out while hidden has zero width - resize on reveal.
    panel.querySelectorAll('.js-plotly-plot').forEach(function (gd) {{
      if (window.Plotly) {{ window.Plotly.Plots.resize(gd); }}
    }});
  }});
}});
</script>
</body>
</html>
"""


PLACEHOLDER = (
    '<div class="placeholder">Forecast chart goes here.<br>'
    'Set <code>FORECAST_CSV</code> at the top of the script.</div>'
)


def _find_plot_function(mod):
    """Find the function in the module that builds the figure."""
    if FORECAST_FUNC:
        return getattr(mod, FORECAST_FUNC)
    for name in ("plot_forecast_plotly", "build_figure", "make_figure",
                 "plot_forecast", "create_figure", "plot"):
        if callable(getattr(mod, name, None)):
            return getattr(mod, name)
    cands = [f for n, f in vars(mod).items()
             if callable(f) and not n.startswith("_")
             and getattr(f, "__module__", "") == mod.__name__
             and ("plot" in n or "fig" in n)]
    if len(cands) == 1:
        return cands[0]
    raise AttributeError(
        f"could not decide which function to call in {FORECAST_MODULE}.py. "
        f"Set FORECAST_FUNC to one of: "
        f"{[n for n, f in vars(mod).items() if callable(f) and not n.startswith('_')]}"
    )


def forecast_fragment() -> str:
    """Build the forecast figure and return it as an embeddable fragment."""
    if not FORECAST_CSV or not Path(FORECAST_CSV).exists():
        print(f"  no forecast CSV at {FORECAST_CSV} - placeholder kept")
        return PLACEHOLDER
    try:
        mod = importlib.import_module(FORECAST_MODULE)
        fn = _find_plot_function(mod)
    except (ImportError, AttributeError) as e:
        print(f"  cannot use {FORECAST_MODULE}.py: {e}")
        print(f"    looked in: {HERE}")
        print(f"    .py files there: {sorted(f.name for f in HERE.glob('*.py'))}")
        return PLACEHOLDER

    df = pd.read_csv(FORECAST_CSV)

    # Pass only the arguments the function actually declares, so this keeps
    # working if its signature changes.
    import inspect
    params = inspect.signature(fn).parameters
    kwargs = {}
    for key, val in (("sample_id", FORECAST_SAMPLE), ("sample", FORECAST_SAMPLE),
                     ("station", FORECAST_STATION), ("show", False)):
        if key in params:
            kwargs[key] = val

    # A function that calls fig.show() itself would open a browser tab per run.
    original_show = go.Figure.show
    go.Figure.show = lambda self, *a, **k: None
    try:
        fc = fn(df, **kwargs)
    except TypeError:
        fc = fn(FORECAST_CSV, **kwargs)      # maybe it wants a path, not a frame
    finally:
        go.Figure.show = original_show

    if not isinstance(fc, go.Figure):
        print(f"  {fn.__name__}() returned {type(fc).__name__}, not a Figure.")
        print("    Add `return fig` at the end of that function.")
        return PLACEHOLDER

    # A hardcoded width forces horizontal scrolling inside the tab.
    fc.update_layout(width=None, autosize=True, height=560)

    print(f"  forecast figure embedded via {fn.__name__}()")
    return ('<div class="plotwrap">'
            + fc.to_html(full_html=False, include_plotlyjs=False,
                         config=dict(displayModeBar=False, responsive=True))
            + "</div>")


def kpi_html(items) -> str:
    out = []
    for k in items:
        tone = k.get("tone", "")
        out.append(
            f'<div><div class="kpi-label">{k["label"]}</div>'
            f'<div class="kpi-value {tone}">{k["value"]}</div>'
            f'<div class="kpi-note">{k.get("note", "")}</div></div>'
        )
    return "\n".join(out)


def write_page(fig: go.Figure, comp: str) -> str:
    """Write a complete HTML page: headline, the figure, and a caption."""
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        auto_play=False,
        post_script=LOOP_JS,
        config=dict(displayModeBar=False, responsive=True),
    )
    name = f"wind_field_{comp}.html"
    with open(name, "w", encoding="utf-8") as fh:
        fh.write(PAGE_TEMPLATE.format(
            headline=PAGE_HEADLINE,
            subtitle=f"{PAGE_SUBTITLE} &middot; {LABELS[comp]}",
            kpis=kpi_html(KPIS),
            forecast=forecast_fragment(),
            plot=plot_html,
        ))
    return name


if __name__ == "__main__":
    for comp in COMPONENTS:
        f = build_figure(comp)
        print(f"  wrote {write_page(f, comp)}")
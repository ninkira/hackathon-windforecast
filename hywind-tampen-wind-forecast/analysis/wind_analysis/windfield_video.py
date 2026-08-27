"""
Render the wind field as a real video file (MP4 + GIF).

Why not the browser: a Plotly heatmap frame needs a full redraw, and during
that redraw the canvas is blank. In a screen recording of the HTML version,
196 of 286 frames came out white. A rendered video has none of that - every
frame is drawn once, offline, and plays back perfectly.

Put this next to windfield_animation.py and run:  python windfield_video.py
Output: windfield.mp4 and windfield.gif
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

import windfield_animation as wa

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
N_EPISODES = 40          # how many episodes to walk through
SUBFRAMES = 8            # interpolated frames between episodes = smoothness
FPS = 20                 # playback rate of the video
OUT_WIDTH = 1000         # px

ARROW_EVERY = 7         # higher = fewer arrows
ARROW_SCALE = 120         # LOWER = LONGER arrows
ARROW_WIDTH = 0.0032     # shaft thickness
ARROW_HEADWIDTH = 3.2    # head width, in multiples of shaft width
ARROW_HEADLENGTH = 5.0   # head length
ARROW_COLOR = "#111111"
ARROW_ALPHA = 0.9
ARROW_PIVOT = "middle"   # "tail" | "middle" | "tip"

CMAP = "YlGnBu"          # "YlGnBu", "viridis", "turbo", "jet"
FIELD_ALPHA = 0.55
CLIP_PCT = 98

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------
# Build the sequence of fields
# ---------------------------------------------------------------
long = wa.long
eps = sorted(long[wa.EPISODE_COL].unique(), key=wa.episode_num)[:N_EPISODES]
print(f"{len(eps)} episodes: {eps[0]} .. {eps[-1]}")

raw = []
for e in eps:
    g = long[long[wa.EPISODE_COL] == e]
    _, gu, gv = wa.interpolate_field(g, wa.gx, wa.gy, wa.STATION_POS)
    raw.append((gu, gv))

# Blend the COMPONENTS between episodes, never the speed: a vector turning
# from east to north must pass through north-east at constant length, not
# shrink to zero and grow back.
frames = []
stamps = []
for i in range(len(raw) - 1):
    (u0, v0), (u1, v1) = raw[i], raw[i + 1]
    for k in range(SUBFRAMES):
        a = k / SUBFRAMES
        frames.append(((1 - a) * u0 + a * u1, (1 - a) * v0 + a * v1))
        stamps.append(eps[i] if k == 0 else f"{eps[i]}\u2192{eps[i+1]}")
frames.append(raw[-1])
stamps.append(eps[-1])
print(f"{len(frames)} rendered frames at {FPS} fps "
      f"= {len(frames) / FPS:.1f} s")

speeds = [np.hypot(u, v) for u, v in frames]
vmax = float(np.percentile(np.concatenate([s.ravel() for s in speeds]), CLIP_PCT))
print(f"colour scale 0 - {vmax:.1f} m/s")

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
W, H = wa.W, wa.H
dpi = 100
fig, ax = plt.subplots(figsize=(OUT_WIDTH / dpi, OUT_WIDTH * H / W / dpi), dpi=dpi)
fig.subplots_adjust(left=0, right=0.88, top=1, bottom=0)

ax.imshow(wa.img, extent=[0, W, H, 0], zorder=0)
im = ax.imshow(speeds[0], extent=[0, W, H, 0], origin="upper",
               cmap=CMAP, vmin=0, vmax=vmax, alpha=FIELD_ALPHA,
               interpolation="bilinear", zorder=1)

s = ARROW_EVERY
GX, GY = wa.GX[::s, ::s], wa.GY[::s, ::s]
u0, v0 = frames[0]
q = ax.quiver(GX, GY, u0[::s, ::s], -v0[::s, ::s],   # -v: image y points down
              scale=ARROW_SCALE, width=ARROW_WIDTH,
              color="#111111", zorder=3)

for name, (px, py) in wa.STATIONS.items():
    ax.plot(px, py, "o", mfc="none", mec="#111111", mew=1.6, ms=6, zorder=4)
    ax.text(px + 8, py + 4, name, fontsize=7, color="#111111", zorder=4)

stamp = ax.text(W * 0.02, H * 0.05, stamps[0], fontsize=14, color="#111111",
                va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.75))

cax = fig.add_axes([0.90, 0.12, 0.022, 0.76])
cb = fig.colorbar(im, cax=cax)
cb.set_label("m/s", rotation=0, labelpad=12)

ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def update(i):
    u, v = frames[i]
    im.set_data(speeds[i])
    q.set_UVC(u[::s, ::s], -v[::s, ::s])
    stamp.set_text(stamps[i])
    return im, q, stamp


anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, blit=False)

mp4 = HERE / "windfield.mp4"
try:
    anim.save(mp4, writer=FFMpegWriter(fps=FPS, bitrate=3200))
    print(f"wrote {mp4}")
except Exception as e:
    print(f"no mp4 ({type(e).__name__}: {e}) - ffmpeg missing?")

gif = HERE / "windfield.gif"
anim.save(gif, writer=PillowWriter(fps=FPS))
print(f"wrote {gif}")
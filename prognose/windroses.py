import MLForecast.sources
import MLForecast.preprocessing as prep
import MLForecast.visualize
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

from windrose import plot_windrose, WindroseAxes
import numpy as np

cwd = pathlib.Path.cwd()
datadir = cwd / "data"


sns.set_theme("paper", font="NewComputerModern", style="ticks")
colormap = plt.get_cmap("magma_r")

FONT_SIZE = 10
plt.rc('font', size=FONT_SIZE)
plt.rc('axes', titlesize=FONT_SIZE)
plt.rc('axes', labelsize=FONT_SIZE)
plt.rc('xtick', labelsize=FONT_SIZE)
plt.rc('ytick', labelsize=FONT_SIZE)
plt.rc('figure', titlesize=FONT_SIZE)

fig = plt.figure(figsize=(6.4, 7))
rects = [
    [0.05, 0.1, 0.4, 0.4],
    [0.55, 0.1, 0.4, 0.4],
    [0.05, 0.55, 0.4, 0.4],
    [0.55, 0.55, 0.4, 0.4],
]
anno_positions = [
    [0.05, 0.51],
    [0.55, 0.51],
    [0.05, 0.96],
    [0.55, 0.96],
]
annos = [
    "Günzburg",
    "Laupheim",
    "Münsingen-\nApfelstetten",
    "Stötten"
]

station_ids = ["01886", "02886", "03402", "04887"]


for station_id, rect, anno_position, anno in zip(station_ids, rects, anno_positions, annos):

    filename = f"{station_id}.pkl"

    df = MLForecast.sources.load_merged_dataframe(datadir, [filename])
    #df = df.head(10)

    df = df[[f"WS_{station_id}", f"WD_{station_id}"]]
    df = df.dropna()
    df = df.rename(columns={f"WS_{station_id}": "speed",
                   f"WD_{station_id}": "direction"})

    ws = df["speed"]
    wd = df["direction"]

    ax = WindroseAxes.from_ax(fig=fig, rect=rect, theta_labels=[
                              "O", "NO", "N", "NW", "W", "SW", "S", "SO"])
    ax.bar(wd, ws, normed=True, opening=0.7, bins=np.arange(
        0, 8, 1), edgecolor='black', cmap=colormap)

    tick_range = np.arange(5, 25+1, step=5)
    ax.set_yticks(tick_range)
    ax.set_yticklabels([f"{i}%" for i in tick_range])

    plt.annotate(anno, anno_position,
                 xycoords='figure fraction', weight='bold')

ax.set_legend()
legend = ax.get_legend()
ax._remove_legend(legend)
labels = [f"{t.get_text()} m/s" for t in legend.get_texts()]
ncol = int(len(legend.legendHandles)/2)
figlegend = fig.legend(legend.legendHandles, labels,
                       loc="lower center", ncol=ncol,
                       frameon=False)
for text in figlegend.get_texts():
    text.set_fontsize(FONT_SIZE)

plt.savefig("windroses.pdf", dpi=600)
plt.show()
print("finish")

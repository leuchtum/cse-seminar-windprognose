import numpy as np
import MLForecast.sources
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


def make_new_ticks(ticks):
    return_ticks = []
    for tick in ticks:
        if "RH_" in tick:
            return_ticks.append("$RH$")
        elif "SD_" in tick:
            return_ticks.append("$SD$")
        elif "T_" in tick:
            return_ticks.append("$T$")
        elif "WD_" in tick:
            return_ticks.append("$\\varphi$")
        elif "WS_" in tick:
            return_ticks.append("$v$")
        elif "_SEAS" in tick:
            return_ticks.append("$cal_{seas}$")
        elif "_HOUR" in tick:
            return_ticks.append("$cal_{day}$")
        elif "P_" in tick:
            return_ticks.append("$p$")
        else:
            raise ValueError(f"Key {tick} unknown")
    return return_ticks


# Load data
cwd = pathlib.Path.cwd()
datadir = cwd / "data"
filenames = ["04887.pkl", "calendar_data.pkl"]
df = MLForecast.sources.load_merged_dataframe(datadir, filenames)

# Make correlation
df = df.loc[df.index < datetime(2021, 1, 1)]  # Crop, so there is no bias
corr = df.corr()

# Drop not needed columns, generate mask
plot_corr = corr.drop(corr.keys()[-1], axis=1).drop(corr.keys()[0], axis=0)
mask = np.triu(np.ones_like(plot_corr, dtype=bool), k=1)

# Make new tick labels
xticks = make_new_ticks(plot_corr.keys())
yticks = make_new_ticks(plot_corr.index)

# Calc limits for colorbar
max_mask = np.tril(np.ones(len(df.keys()), dtype=bool), k=-1)
vmax = np.around(corr.where(max_mask).max().max(), decimals=1)
vmin = np.around(corr.where(max_mask).min().min(), decimals=1)

# Define colormap
cmap = sns.diverging_palette(2.05, 99.66, s=50, as_cmap=True)

# Actual plot
plt.rcParams['mathtext.fontset'] = 'cm'
f, ax = plt.subplots(figsize=(6, 4))
sns.set_theme(style="white", font='NewComputerModern')
sns.set_context(
    "paper",
    rc={
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10
    }
)
sns.heatmap(
    plot_corr,
    mask=mask,
    cmap=cmap,
    vmax=vmax,
    vmin=vmin,
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": .8},
    annot=True,
    fmt=".3f",
    yticklabels=yticks,
    xticklabels=xticks
)

# Post cleanup
plt.yticks(rotation=0)
plt.subplots_adjust(top=0.95)
plt.savefig("corr.pdf", dpi=600)
plt.show()


print("finished")

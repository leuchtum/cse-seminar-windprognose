import MLForecast.sources
import MLForecast.preprocessing as prep
import MLForecast.visualize
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt


cwd = pathlib.Path.cwd()
datadir = cwd / "data"
filenames = ["04887.pkl", "calendar_data.pkl"]
df = MLForecast.sources.load_merged_dataframe(datadir, filenames)
#df = df.head(800)

df = df.dropna()

df_ws_wd = df[["WS_04887", "WD_04887"]]
df_wx_wy = prep.pol2cart(df)

sns.set(font="NewComputerModern")
sns.set_context("paper", rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":10})

fig, axes = plt.subplots(2,1)
colormap=plt.get_cmap("magma_r")

g_left_1 = sns.kdeplot(x=df_ws_wd["WD_04887"], y=df_ws_wd["WS_04887"], ax=axes[0], fill=True, levels=100, cmap=colormap, thresh=0)
g_left_2 = sns.kdeplot(x=df_ws_wd["WD_04887"], y=df_ws_wd["WS_04887"], ax=axes[0], levels=10, color="k", thresh=.2)
g_right_1 = sns.kdeplot(x=df_wx_wy["WX_04887"], y=df_wx_wy["WY_04887"], ax=axes[1], fill=True, levels=100, cmap=colormap, thresh=0)
g_right_2 = sns.kdeplot(x=df_wx_wy["WX_04887"], y=df_wx_wy["WY_04887"], ax=axes[1], levels=10, color="k", thresh=.2)

axes[0].set_xlabel(r"Windrichtung in $^\circ$")
axes[0].set_ylabel(r"Windgeschwindigkeit in m/s")
xlim = [0,360]
ylim = [0,12]
g_left_1.set(xlim = xlim, ylim=ylim)

axes[1].set_xlabel(r"Windgeschwindigkeit in m/s")
axes[1].set_ylabel(r"")
xlim = [-12,9]
ylim = [-7,5]
axes[1].plot(xlim,[0,0], "k", linestyle='dashed')
axes[1].plot([0,0],ylim, "k", linestyle='dashed')
g_right_1.set(xlim = xlim, ylim=ylim)
#plt.axis('square')
#import numpy as np
#asp = np.diff(axes[1].get_xlim())[0] / np.diff(axes[1].get_ylim())[0]
#axes[1].set_aspect(asp)
plt.subplots_adjust(top=0.95)
plt.savefig("pol2cart2.pdf",dpi=600)
plt.show()
print("finished")
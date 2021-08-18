import MLForecast.sources
import MLForecast.preprocessing as prep
import MLForecast.visualize
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd

PATH = "/home/daniel/Dokumente/RNNseq2vec_HORI12_HIST144_POLxy_SINGLE0/"
INDENT = "RNNseq2vec_HORI12_HIST144_POLxy_SINGLE0"
SEP = "_"
    
def load_from_others(name):
    filename = PATH + INDENT + "_" + name + ".pkl"
    with open(filename, "rb") as loadfile:
        print(f"READ IN {filename}")
        f = pickle.load(loadfile)
    return f

filename = PATH + INDENT + "_" + "dwddf" + ".pkl"
df = pd.read_pickle(filename)[["WS_01886", "WD_01886"]]

#df = df.head(3000)

df = df.dropna()

df_ws_wd = df[["WS_01886", "WD_01886"]]
df_wx_wy = prep.pol2cart(df)

sns.set(font="NewComputerModern")
sns.set_context("paper", rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":10})

fig, axes = plt.subplots(2,1, figsize=(5,6))
colormap=plt.get_cmap("magma_r")

g_left_1 = sns.kdeplot(x=df_ws_wd["WD_01886"], y=df_ws_wd["WS_01886"], ax=axes[0], fill=True, levels=100, cmap=colormap, thresh=0)
g_left_2 = sns.kdeplot(x=df_ws_wd["WD_01886"], y=df_ws_wd["WS_01886"], ax=axes[0], levels=10, color="k", thresh=.2)
g_right_1 = sns.kdeplot(x=df_wx_wy["WX_01886"], y=df_wx_wy["WY_01886"], ax=axes[1], fill=True, levels=100, cmap=colormap, thresh=0)
g_right_2 = sns.kdeplot(x=df_wx_wy["WX_01886"], y=df_wx_wy["WY_01886"], ax=axes[1], levels=10, color="k", thresh=.2)

axes[0].set_xlabel(r"Windrichtung in $^\circ$")
axes[0].set_ylabel(r"Windgeschwindigkeit in m/s")
xlim = [0,360]
ylim = [0,12]
g_left_1.set(xlim = xlim, ylim=ylim)

axes[1].set_xlabel(r"Windgeschwindigkeit in m/s")
axes[1].set_ylabel(r"Windgeschwindigkeit in m/s")
xlim = [-12,9]
ylim = [-7,5]
axes[1].plot(xlim,[0,0], "k", linestyle='dashed')
axes[1].plot([0,0],ylim, "k", linestyle='dashed')
g_right_1.set(xlim = xlim, ylim=ylim)
#plt.axis('square')
#import numpy as np
#asp = np.diff(axes[1].get_xlim())[0] / np.diff(axes[1].get_ylim())[0]
#axes[1].set_aspect(asp)
#plt.subplots_adjust(top=0.98)
plt.tight_layout()
plt.savefig("pol2cart2.pdf",dpi=600)
plt.show()
print("finished")
import MLForecast.sources
import MLForecast.preprocessing as prep
import MLForecast.visualize
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt


cwd = pathlib.Path.cwd()
datadir = cwd / "data"
filenames = ["15444.pkl", "calendar_data.pkl"]
df = MLForecast.sources.load_merged_dataframe(datadir, filenames)
df = df.head(4000)

df.dropna(axis=1)

df_ws_wd = df[["WS_15444", "WD_15444"]]
df_wx_wy = prep.pol2cart(df)

mymap=sns.light_palette("seagreen", as_cmap=True)

f, ax = plt.subplots(figsize=(6, 6))
#sns.scatterplot(x=df_ws_wd["WD_15444"], y=df_ws_wd["WS_15444"], s=5, color=".15")
#sns.histplot(x=df_ws_wd["WD_15444"], y=df_ws_wd["WS_15444"], cmap="mako")
sns.kdeplot(x=df_ws_wd["WD_15444"], y=df_ws_wd["WS_15444"], fill=True, levels=20)
#sns.histplot(x=df_ws_wd["WD_15444"], y=df_ws_wd["WS_15444"], cbar=True, cbar_kws=dict(shrink=.75))
#sns.displot(x=df_ws_wd["WD_15444"], y=df_ws_wd["WS_15444"], kind="kde")
plt.show()
print("finished")

import pandas as pd
import numpy as np
from datetime import datetime


def get_day_sin(dates):
    vals = np.zeros(len(dates))
    for i in range(len(dates)):
        val = dates[i].hour
        vals[i] = (np.cos(np.pi* (val / 12 + 1)) + 1)/2
    return vals

def get_seas_sin(dates):
    vals = np.zeros(len(dates))
    for i in range(len(dates)):
        val = dates[i].day_of_year - 1 + dates[i].hour / 24
        vals[i] = (np.cos(np.pi* (val / 182.5 + 1)) + 1)/2
    return vals

calindex = pd.date_range(start=datetime(2016,1,1), end=datetime.now() , freq="H")

hours = get_day_sin(calindex.to_list())
seasons = get_seas_sin(calindex.to_list())

df = pd.DataFrame([seasons, hours]).T
df.columns = ["CAL_COS_SEAS", "CAL_COS_HOUR"]
df.index = calindex

df.to_pickle(__file__.replace(".py", ".pkl"))
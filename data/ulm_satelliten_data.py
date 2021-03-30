import pandas as pd
from datetime import timedelta

# Get path + filename without suffix
filename = __file__.replace(".py", "")

# Read in
df = pd.read_excel(f"{filename}.xlsx")

# Make new index from column "Date"  and "Hour"
hours = [timedelta(hours=i) for i in (df.Hour - 1).to_list()] # minus 1, because 1to24 instead of 0to23
dates = df.Date.to_list()
new_datetimes = []
for h, d in zip(hours, dates):
    new_datetimes.append(d+h)
df.index = new_datetimes

# Drop irrelevant columns and rename remaining columns
df = df.drop(["Date", "Hour", "Rainfall", "Max incomming solar irradiation"], axis=1)
df = df.rename(
    columns={
        "Temperature": "T",
        "Humidity": "RH",
        "Pressure": "P",
        "Wind Speed": "WS",
        "Wind Direction": "WD",
        "Global Irradiation": "GHI"
        }
    )

# Save to pickle
df.to_pickle(f"{filename}.pkl")
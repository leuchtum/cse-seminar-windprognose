import numpy as np
import pandas as pd
import os
import tempfile
import zipfile
from datetime import datetime


drop_before = datetime(2016,1,1)
rename_lookup = {
    "TT_TU": "T",
    "RF_TU": "RH",
    "F": "WS",
    "D": "WD",
    "SD_SO": "SD",
    "P0": "P"
}

# Iterate over all files
stations_historical = {}
stations_recent = {}
path = __file__.replace("zip2pkl.py", "")
for filename in os.listdir(path):
    # start, if file is zip
    if ".zip" in filename:
        station_id = filename.split("_")[2]
        
        print(f"READ IN: ID={station_id} file={filename}")
        
        # Make temporary directory
        with tempfile.TemporaryDirectory() as tmppath:
            # Unzip into temporary directory
            with zipfile.ZipFile(path+filename, 'r') as zip_ref:
                zip_ref.extractall(tmppath)
            # Iterate over filenames, read data into DataFrame
            for tmpfilename in os.listdir(tmppath):
                if "produkt" in tmpfilename:
                    f = f"{tmppath}/{tmpfilename}"
                    df = pd.read_csv(f, delimiter=";")

        # Make datetime index
        timestamps = pd.to_datetime(df.MESS_DATUM, format="%Y%m%d%H").to_list()
        df.index = timestamps
        new_index = pd.date_range(start=timestamps[0], end=timestamps[-1], freq="H")
        empty_df = pd.DataFrame(index=new_index)
        df = pd.concat([df, empty_df], axis=1)
        
        # drop columns that are not needed
        todrop = ["STATIONS_ID", "MESS_DATUM", "eor"]
        for key in df.keys():
            if "QN_" in key:
                todrop.append(key)
            if "P" in key:
                if "P0" in key:
                    continue
                todrop.append(key)
        df = df.drop(todrop, axis=1)
        
        # drop rows that are to old
        todrop = (df.index >= drop_before)
        df = df.loc[todrop]
        
        # Rename columns
        new_names = {}
        for key in df.keys():
            clean_key = key.replace(" ", "")
            if clean_key in rename_lookup:
                new_names[key] = rename_lookup[clean_key]
            else:
                new_names[key] = clean_key
        
        for key, clean_key in new_names.items():
            new_names[key] = f"{clean_key}_{station_id}"
            
        df = df.rename(columns=new_names)
        
        # Make -999 to NaN
        df = df.replace(-999,np.nan)
        
        # When it is night, SD will not be written. Write 0 instead of NaN if night
        if any(["SD" in key for key in df.keys()]):
            for key in df.keys():
                if key[:2] == "SD":
                    sd_key = key
            for i in range(len(df[sd_key])):
                idx = df[sd_key].index[i]
                if idx.hour in [21,22,23,0,1,2]:
                    df[sd_key].values[i] = 0
        
        # Append stations with station_id as key
        if "akt" in filename:
            if station_id not in stations_recent:
                stations_recent[station_id] = []
            stations_recent[station_id].append(df)
        if "hist" in filename:
            if station_id not in stations_historical:
                stations_historical[station_id] = []
            stations_historical[station_id].append(df)

# Concat measurements to one DataFrame
stations_recent_dfs = {}    
stations_historical_dfs = {}
for station_id in stations_recent:
    df_list = stations_recent[station_id]
    stations_recent_dfs[station_id] = pd.concat(df_list, axis=1)
for station_id in stations_historical:
    df_list = stations_historical[station_id]
    stations_historical_dfs[station_id] = pd.concat(df_list, axis=1)

# Combine recent and historical
stations = {}
for station_id in stations_historical_dfs:
    recent = stations_recent_dfs[station_id]
    historical = stations_historical_dfs[station_id]
    stations[station_id] = historical.combine_first(recent)

# Save to pickle
for station_id, df in stations.items():
    print(f"SAVE: {station_id}.pkl")
    df.to_pickle(f"{path}{station_id}.pkl")
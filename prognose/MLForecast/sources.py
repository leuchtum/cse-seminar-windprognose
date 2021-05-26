import pathlib
import requests
import numpy as np
import pandas as pd
import os
import tempfile
import zipfile
from datetime import datetime
from tqdm import tqdm

TEMP_DIR_NAME = "tmp_dwd_files"


def load_merged_dataframe(path, filenames):
    """Load and merge DataFrames form pickle.

    Args:
        path (pathlib.Path): Directory containing the Data Pickles.
        filenames (list): List containing the filenames that will be loaded and merged.

    Returns:
        pd.DataFrame: Merged DataFrame
    """

    dfs = [pd.read_pickle(path / f) for f in filenames]
    return pd.concat(dfs, axis=1)


def get_temp_dir():
    tmp = pathlib.Path(tempfile.gettempdir())
    path = tmp / TEMP_DIR_NAME

    if not path.exists():
        path.mkdir()

    return path


def join_urls(urls, suburls):
    final_urls = []
    for url in urls:
        for suburl in suburls:
            final_urls.append(f"{url}/{suburl}")
    return final_urls


def get_file(url, directory):
    file_name_start_pos = url.rfind("/") + 1
    path = directory / url[file_name_start_pos:]

    if not path.exists():
        download_from_url(url, path)

    return path


def download_from_url(url, path):
    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(path, 'wb') as f:
            for data in r:
                f.write(data)


def scrap_decode(content):
    file_names = []
    for line in content.splitlines():
        if "<a href=" in line.split('"'):
            file_names.append(line.split('"')[1])

    return file_names


def scrap_dwd_urls(baseurl, root_recursive=True):
    if root_recursive:
        print(f"CHECK FOR NEW FILES AT {baseurl}")

    final_urls = []
    r = requests.get(baseurl, stream=True)
    if r.status_code == requests.codes.ok:
        suburls = scrap_decode(r.content.decode())
        for suburl in suburls:
            if suburl[-1] == "/":
                final_urls.extend(scrap_dwd_urls(
                    baseurl + suburl, root_recursive=False))
            else:
                final_urls.append(baseurl + suburl)

    return final_urls


def read_in_from_zip_to_df(file_path):
    with tempfile.TemporaryDirectory() as tmp_path:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_path)
        for tmpfilename in os.listdir(tmp_path):
            if "produkt" in tmpfilename:
                f = f"{tmp_path}/{tmpfilename}"
                return pd.read_csv(f, delimiter=";")


def clean_up_df(df, station_id):
    # remake index
    timestamps = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H").to_list()
    df.index = timestamps
    new_index = pd.date_range(
        start=timestamps[0], end=timestamps[-1], freq="H")
    empty_df = pd.DataFrame(index=new_index)
    df = pd.concat([df, empty_df], axis=1)

    # drop columns that are not needed
    todrop = [
        "MESS_DATUM",
        "STATIONS_ID",
        "eor",
        *[i for i in df.keys() if "QN_" in i or "P0" in i],
    ]
    df = df.drop(columns=todrop)

    # Make -999 to NaN
    df = df.replace(-999, np.nan)

    # Rename columns
    rename_lookup = {
        "TT_TU": "T",
        "RF_TU": "RH",
        "F": "WS",
        "D": "WD",
        "SD_SO": "SD",
        "P0": "P"
    }
    new_names = {}
    for key in df.keys():
        clean_key = key.replace(" ", "")
        new_names[key] = rename_lookup.get(clean_key, clean_key)

    new_names = {key: f"{new_names[key]}_{station_id}" for key in new_names}
    df = df.rename(columns=new_names)

    return df


def check_for_particularities(df):
    # When it is night, SD will not be written. Write 0 instead of NaN if night
    if any("SD" in key for key in df.keys()):
        for key in df.keys():
            if key[:2] == "SD":
                sd_key = key
                break
        for i in range(len(df[sd_key])):
            idx = df[sd_key].index[i]
            if idx.hour in [21, 22, 23, 0, 1, 2]:
                df[sd_key].values[i] = 0

    # For wind direction a NaN value is wirtten as 990 instead of -999
    if any("WD" in key for key in df.keys()):
        for key in df.keys():
            if key[:2] == "WD":
                break
        idx = df.index[df[key] == 990.0]
        df.loc[idx] = np.nan

    return df


class DWDStationsHourly():
    def __init__(self, station_ids, observations):

        self.rooturl = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/"
        self.allurls = scrap_dwd_urls(self.rooturl)
        self.directory = get_temp_dir()

        self.station_ids = station_ids
        self.observations = observations

        self.download()
        self.read_in()

    def download(self):
        print("DOWNLOAD FILES")

        station_urls = [i for i in self.allurls if any(
            j in i for j in self.station_ids)]
        self.urls = [i for i in station_urls if any(
            j in i.split("/") for j in self.observations)]

        self.relations = {station_id: [] for station_id in self.station_ids}
        for url in tqdm(self.urls):
            for station_id in self.station_ids:
                if station_id in url:
                    self.relations[station_id].append(
                        get_file(url, self.directory))
                    continue

    def read_in(self):
        print("READ IN")

        recent_dfs = {sid: [] for sid in self.station_ids}
        historical_dfs = {sid: [] for sid in self.station_ids}

        for station_id, file_names in self.relations.items():
            for file_name in tqdm(file_names, desc=station_id):
                df = read_in_from_zip_to_df(self.directory / file_name)
                df = clean_up_df(df, station_id)
                df = check_for_particularities(df)
                if "hist" in str(file_name):
                    historical_dfs[station_id].append(df)
                else:
                    recent_dfs[station_id].append(df)

        recent_dfs = {sid: pd.concat(
            recent_dfs[sid], axis=1) for sid in self.station_ids}
        historical_dfs = {sid: pd.concat(
            historical_dfs[sid], axis=1) for sid in self.station_ids}

        self.dfs = {sid: None for sid in self.station_ids}
        for station_id in self.station_ids:
            h = historical_dfs[station_id]
            r = recent_dfs[station_id]
            self.dfs[station_id] = h.combine_first(r)

    def get_data(self, station_id=None, drop_before=None, drop_after=None):
        if station_id:
            df = self.dfs[station_id]
        else:
            dfs = [self.dfs[sid] for sid in self.dfs]
            df = pd.concat(dfs, axis=1)

        if drop_before is not None:
            df = df.loc[(df.index >= drop_before)]
        if drop_after is not None:
            df = df.loc[df.index <= drop_after]

        return df


class CalendricalDataHourly:
    def __init__(self, start=datetime(1900, 1, 1), end=datetime.now()):

        self.index = pd.date_range(start=start, end=end, freq="H")
        self.cos_hour = self.calc_cos_hour(self.index.to_list())
        self.cos_seas = self.calc_cos_seas(self.index.to_list())

    def get_data(self, cos_hour=True, cos_seas=True):
        data = []
        cols = []

        if cos_hour:
            data.append(self.cos_hour)
            cols.append("CALENDAR_COS_HOUR")
        if cos_seas:
            data.append(self.cos_seas)
            cols.append("CALENDAR_COS_SEAS")

        return pd.DataFrame(np.array(data).T, index=self.index, columns=cols)

    def calc_cos_hour(self, dates):
        vals = np.zeros(len(dates))
        for i in range(len(dates)):
            val = dates[i].hour
            vals[i] = (np.cos(np.pi * (val / 12 + 1)) + 1)/2
        return vals

    def calc_cos_seas(self, dates):
        vals = np.zeros(len(dates))
        for i in range(len(dates)):
            val = dates[i].day_of_year - 1 + dates[i].hour / 24
            vals[i] = (np.cos(np.pi * (val / 182.5 + 1)) + 1)/2
        return vals


if __name__ == "__main__":
    dwd = DWDStationsHourly(["15444", "04887"], [
                            "wind", "air_temperature", "pressure", "sun"])
    df = dwd.get_station_data("15444", drop_before=datetime(2017, 1, 1))

    df2 = CalendricalDataHourly(start=datetime(2014, 1, 1)).get_data()

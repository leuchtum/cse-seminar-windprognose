import pandas as pd
import numpy as np
from tqdm import tqdm


def interpolate_columnwise(df: pd.DataFrame, max_gap=3) -> pd.DataFrame:
    """Fill missing data columnwise by interpolating with the last available data.

    Args:
        df (pd.DataFrame): Input DataFrame
        max_gap (int): Maximal gap per column to interpolate over.

    Returns:
        pd.DataFrame: Output DataFrame with filled data
    """
    df = df.copy()

    for colname in df:
        df[colname] = df[colname].interpolate(limit=max_gap)

    return df


def normalize_columnwise(df: pd.DataFrame, how="minmax", exclude=["CAL_", "TARGET_"]) -> pd.DataFrame:
    """Normalize a DataFrame columnwise.

    Args:
        df (pd.DataFrame): DataFrame to perform normalization
        how (str, optional): Either "minmax" or "znorm". Defaults to "minmax".
        exclude (list, optional): Columns to exclude from normalization.

    Returns:
        pd.DataFrame: Returns the normalized DataFrame
    """
    df = df.copy()

    for colname in df:
        skip = any(ex in colname for ex in exclude)

        if not skip:
            if how == "minmax":
                df[colname] = (df[colname] - df[colname].min()) / \
                    (df[colname].max() - df[colname].min())
            if how == "znorm":
                df[colname] = (df[colname]-df[colname].mean()) / \
                    df[colname].std()

    return df


def pol2cart(df, speed="WS", direction="WD", x="WX", y="WY"):
    """Recode windspeed and direction into x and y component

    Args:
        df (pd.DataFrame): Dataframe containing columns with speed and direction
        direction_name (str, optional): Wind direction column name . Defaults to "WD".
        speed_name (str, optional): Wind speed column name. Defaults to "WS".
        rename (tuple, optional): Columns for wind speed in x and y direction will be named like 
            this. Defaults to ("WX", "WY").
        drop (tuple, optional): Drop's the old columns. Defaults to ("WS", "WD").
    """

    df = df.copy()

    while any("WS" in key for key in df.keys()):
        # get keys
        for key in df.keys():
            if speed in key:
                s = key
            elif direction in key:
                d = key

        wind = df[[s, d]]
        wind = wind.dropna()

        ws = wind[s]
        wd = wind[d]
        wd = wd * np.pi / 180

        wx = ws * np.sin(wd)
        wy = ws * np.cos(wd)

        appendix = ws.name.replace(speed, "")
        assert appendix == wd.name.replace(direction, "")

        wx.name = x + appendix
        wy.name = y + appendix

        df = pd.concat([wx, wy, df], axis=1)
        df = df.drop([d, s], axis=1)

    return df


def cart2pol(df, speed="WS", direction="WD", x="WX", y="WY"):
    df = df.copy()

    while any("WX" in key for key in df.keys()):
        # get keys
        for key in df.keys():
            if x in key:
                xkey = key
            elif y in key:
                ykey = key

        wind = df[[xkey, ykey]]
        wind = wind.dropna()

        wx = wind[xkey]
        wy = wind[ykey]
        
        ws = np.sqrt(wx*wx + wy*wy)
        wd = np.arctan2(-wx,-wy) / np.pi * 180 + 180

        appendix = wx.name.replace(x, "")
        assert appendix == wy.name.replace(y, "")

        ws.name = speed + appendix
        wd.name = direction + appendix

        df = pd.concat([ws, wd, df], axis=1)
        df = df.drop([xkey, ykey], axis=1)

    return df


def split_test_val_train(x, y, split):
    assert sum(split) == 1
    x_test = x[:int(len(x)*split[0])]
    y_test = y[:int(len(y)*split[0])]
    x_val = x[int(len(x)*split[0]):int(len(x)*split[1])]
    y_val = y[int(len(y)*split[0]):int(len(y)*split[1])]
    x_train = x[int(len(x)*split[1]):]
    y_train = y[int(len(x)*split[1]):]
    return (x_test, y_test), (x_val, y_val), (x_train, y_train)


def split_at_gaps(df, windowsize):
    not_nan = df.notna()
    not_nan = not_nan.all(axis=1)
    not_nan.index = range(len(not_nan))

    true_idx = not_nan[not_nan].index

    edges = [true_idx[0]]

    for i in range(len(true_idx) - 1):
        idx = true_idx[i]
        next_idx = true_idx[i+1]
        if next_idx - idx != 1:
            edges.extend([idx, next_idx])

    edges.append(true_idx[-1])

    intervals = []
    for i in range(0, len(edges), 2):
        if edges[i + 1] - edges[i] >= windowsize:
            intervals.append((edges[i], edges[i + 1]))

    return intervals


def sample_sequences(pre_sampling_x, pre_sampling_y, x_width, y_width, shift):
    window_size = max(x_width, y_width + shift)
    assert len(pre_sampling_x) == len(pre_sampling_y)
    intervals = split_at_gaps(pre_sampling_x, window_size)

    # Rename
    pre_sampling_x = pre_sampling_x.rename(
        columns={key: f"INPUT_{key}" for key in pre_sampling_x.keys()})
    pre_sampling_y = pre_sampling_y.rename(
        columns={key: f"LABEL_{key}" for key in pre_sampling_y.keys()})

    # Sample
    x_samples = []
    y_samples = []

    for inter in tqdm(intervals):
        for i in tqdm(range(inter[0], inter[1] - window_size)):
            x_samples.append(pre_sampling_x.iloc[i:i+x_width])
            y_samples.append(pre_sampling_y.iloc[i+shift:i+shift+y_width])

    return x_samples, y_samples


def df_tuple_to_np(df_tuple):
    def convert(df_list): return np.array([d.values for d in df_list])
    return tuple([convert(df_list) for df_list in df_tuple])


class DataBaseReport:
    def __init__(self, pre_sampling_x, pre_sampling_y, post_sampling_x, post_sampling_y, shift, split):
        # Pre sampling
        self.x_names = pre_sampling_x.keys().to_list()
        self.x_stats = pre_sampling_x.describe(
        ).loc[["count", "min", "max", "std"]].to_dict()
        self.y_names = pre_sampling_y.keys().to_list()
        self.y_stats = pre_sampling_y.describe(
        ).loc[["count", "min", "max", "std"]].to_dict()
        self.start = str(pre_sampling_x.index[0])
        self.end = str(pre_sampling_x.index[-1])
        self.n_measurements = len(pre_sampling_x)

        # Post sampling
        self.n_samples = len(post_sampling_x)
        self.x_shape = post_sampling_x[0].shape
        self.y_shape = post_sampling_y[0].shape
        self.split = split
        self.shift = shift

    def report(self):
        return self.__dict__

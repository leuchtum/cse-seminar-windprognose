import pandas as pd
import numpy as np

def load_merged_dataframe(path, filenames):
    """Load and merge DataFrames form pickle.

    Args:
        path (pathlib.Path): Directory containing the Data Pickles.
        filenames (list): List containing the filenames that will be loaded and merged.

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    
    dfs = [pd.read_pickle(path / f) for f in filenames]
    return pd.concat(dfs,axis=1)


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


def normalize_columnwise(df: pd.DataFrame, how="minmax", exclude=["CAL_"]) -> pd.DataFrame:
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
            if how=="minmax":
                df[colname] = (df[colname] - df[colname].min()) / (df[colname].max() - df[colname].min())
            if how=="znorm":
                df[colname] = (df[colname]-df[colname].mean())/df[colname].std()
                
    return df

def split_at_gaps(df, minlength = 24):
    df = df.copy()
    oldindex = df.index
    df.index = range(len(df))
    
    index = df[df.notna().all(axis=1)].index.to_list()
    
    split_at = []
    for i in range(len(index)-1):
        if index[i+1] - index[i] != 1:
            split_at.append(index[i])
    
    dfs = []
    start = 0
    split_at = split_at[::-1]
    if split_at:
        while True:
            end = split_at.pop() + 1
            dfs.append(df.iloc[start:end].dropna())
            start = end + 1
            
            if not split_at:
                dfs.append(df.iloc[start:index[-1]+1].dropna())
                break
    else:
        dfs.append(df.dropna())
        
    for df in dfs:
        df.index = oldindex[df.index]
        
    return [df for df in dfs if len(df)>=minlength]


def pol2cart(df, direction_name="WD", speed_name="WS", rename=("WX", "WY"), drop=("WS", "WD")):
    def inner_pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    df = df.copy()
    
    w = df[[speed_name, direction_name]].dropna()
    
    ws = w[speed_name]
    wd = w[direction_name]
    wd = wd * np.pi / 180
    
    wx = ws * np.cos(wd)
    wy = ws * np.sin(wd)
    
    wx.name = rename[0]
    wy.name = rename[1]
    
    df = pd.concat([df, wx, wy], axis=1)
    for d in drop:
        df = df.drop(d, axis=1)
    return df


def make_targets(df, shift, target_cols=["WX", "WY"]):
    df = df.copy()
    shifted = df[target_cols].shift(-shift)
    
    names = shifted.keys().to_list()
    new_names = {}
    for n in names:
        new_names[n] = f"TARGET_{shift}h_{n}"
    shifted = shifted.rename(columns=new_names)
    
    return pd.concat([df,shifted], axis=1)
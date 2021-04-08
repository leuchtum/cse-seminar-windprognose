from preprocessing import load_merged_dataframe, interpolate_columnwise, normalize_columnwise
import pathlib

if __name__ == '__main__':
    
    cwd = pathlib.Path.cwd()
    datadir = cwd / "data"
    
    filenames = ["15444.pkl", "calendar_data.pkl"]
    
    df = load_merged_dataframe(datadir, filenames)
    df = interpolate_columnwise(df, max_gap=3)
    df = normalize_columnwise(df)
    
    # df = df.dropna()
    # TODO: Target erzeugen, also Spalte WS kopieren, concatinaten und dann shiften
    # df.WS = df.WS.shift(-10)
    print("finish")
import MLForecast.sources
import MLForecast.preprocessing as prep
import MLForecast.normalize
import MLForecast.visualize

from datetime import datetime
import pandas as pd
import numpy as np
import pickle

if __name__ == '__main__':
    PATH = "/home/daniel/Dokumente/"
    SEP = "_"
    
    START = datetime(2016, 1, 1)
    END = datetime(2021, 1, 1)
    X_WIDTH = 144
    Y_WIDTH = 12
    SHIFT = X_WIDTH
    INTERPOLATE = 4
    SPLIT = (.05, .3, .65)  # TEST, VAL, TRAIN
    
    POL = "s" # xy, sd, s for wind xy, speed & dir, speed only
    if POL == "xy":
        LABEL_COLUMNS = ["WX_01886", "WY_01886"]
    elif POL == "sd":
        LABEL_COLUMNS = ["WS_01886", "WD_01886"]
    elif POL == "s":
        LABEL_COLUMNS = ["WS_01886"]
        
    SINGLE = 0
    if SINGLE:
        stations = ["01886"]
    else:
        stations = ["01886", "02886", "03402", "04887"]
    
    INDENT = "_".join(["RNNseq2vec",f"HORI{Y_WIDTH}",f"HIST{X_WIDTH}",f"POL{POL}",f"SINGLE{SINGLE}"])


    print("LOAD")
    dwd = MLForecast.sources.DWDStationsHourly(stations, ["wind", "air_temperature", "pressure", "sun"])
    df_dwd = dwd.get_data(drop_before=START, drop_after=END)
    cal = MLForecast.sources.CalendricalDataHourly(start=START, end=END)
    df_cal = cal.get_data()

    df = pd.concat([df_dwd, df_cal], axis=1)

    print("INTERPOLATE AND POL2CART")
    df = prep.interpolate_columnwise(df, INTERPOLATE)
    if POL == "xy":
        df = prep.pol2cart(df)

    print("FIT NORMALIZER")
    spezial_borders = {
        "WX": (-1, 1),
        "WY": (-1, 1)
    }
    exclude = ["CAL"]
    norm = MLForecast.normalize.MinMaxNormalizer(
        spezial_border=spezial_borders, exclude=exclude)
    norm.fit(df)

    print("MAKE X_DF AND Y_DF")
    df_x = norm.normalize(df)
    df_y = df[LABEL_COLUMNS]

    print("SAMPLE")
    x, y = prep.sample_sequences(df_x, df_y, X_WIDTH, Y_WIDTH, SHIFT)

    database_report = prep.DataBaseReport(
        df_x, df_y, x, y, SHIFT, SPLIT).report()

    print("SPLIT")
    test_df, val_df, train_df = prep.split_test_val_train(x, y, SPLIT)

    test_np = prep.df_tuple_to_np(test_df)
    val_np = prep.df_tuple_to_np(val_df)
    train_np = prep.df_tuple_to_np(train_df)

    print("SAVE")   
    get_name = lambda x, suf: PATH + SEP.join([INDENT,x]) + suf
    
    def save_pandas_data(df, name):
        df.to_pickle(get_name(name, ".pkl"))
        
    def save_numpy_data(array, name):
        with open(get_name(name, ".npy"), 'wb') as f:
            np.save(f, array)
            
    def save_other_data(obj, name):
        with open(get_name(name, ".pkl"), 'wb') as f:
            pickle.dump(obj, f, protocol=4)
      
    save_other_data(df_dwd, "dwddf")      
    save_other_data(df, "rootdf")
    save_other_data(test_np, "testnp")
    save_other_data(val_np, "valnp")
    save_other_data(train_np, "trainnp")
    save_other_data(test_df, "testdf")
    save_other_data(val_df, "valdf")
    save_other_data(train_df, "traindf")
    save_other_data(database_report, "report")
    
    

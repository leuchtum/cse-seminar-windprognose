import MLForecast.sources
import MLForecast.preprocessing as prep
import MLForecast.normalize
import MLForecast.networks
import MLForecast.postprocessing
import MLForecast.visualize
from datetime import datetime
import pandas as pd
import wandb


if __name__ == '__main__':
    X_WIDTH = 200
    Y_WIDTH = 12
    SHIFT = 200
    INTERPOLATE = 4
    SPLIT = (.05, .3, .65) # TEST, VAL, TRAIN
    LABEL_COLUMNS = ["WX_01886", "WY_01886"]
    EPOCHS = 6
    EARLY_STOP = 2
    
    print("LOAD")
    dwd = MLForecast.sources.DWDStationsHourly(["01886", "02886", "03402", "04887"], ["wind", "air_temperature", "pressure", "sun"])
    df_dwd = dwd.get_data(drop_before=datetime(2010,1,1))
    cal = MLForecast.sources.CalendricalDataHourly(start=datetime(2010,1,1))
    df_cal = cal.get_data()
    
    df = pd.concat([df_dwd, df_cal], axis=1)
    
    #df = load_merged_dataframe(datadir, filenames)
    #cwd = pathlib.Path.cwd()
    #datadir = cwd / "data"
    #filenames = ["15444.pkl", "calendar_data.pkl"]
    #df = df.head(4000)
    
    print("INTERPOLATE AND POL2CART")
    df = prep.interpolate_columnwise(df, INTERPOLATE)
    df =  prep.pol2cart(df)
    
    print("FIT NORMALIZER")
    spezial_borders = {
        "WX": (-1, 1),
        "WY": (-1, 1)
    }
    exclude = ["CAL"]
    norm = MLForecast.normalize.MinMaxNormalizer(spezial_border=spezial_borders, exclude=exclude)
    norm.fit(df)
    
    print("MAKE X_DF AND Y_DF")
    df_x = norm.normalize(df)
    df_y = df[LABEL_COLUMNS]
    
    print("SAMPLE")
    x, y =  prep.sample_sequences(df_x, df_y, X_WIDTH, Y_WIDTH, SHIFT)
    #plot_input_label_example(x[30], y[30])
    
    database_report =  prep.DataBaseReport(df_x, df_y, x, y, SHIFT, SPLIT).report()
    
    print("SPLIT")
    test, val, train =  prep.split_test_val_train(x, y, SPLIT)
    
    test_np =  prep.df_tuple_to_np(test)
    val_np =  prep.df_tuple_to_np(val)
    train_np =  prep.df_tuple_to_np(train)
    
    
    
    structure = ["LSTM_x", "DROP_d", "DENSE_y", "OUT"]
    network_config = {
        #"LSTM": {
        #    "recurrent_regularizer": "L1_0.0001"
        #},
        "x": 64,
        "y": 64,
        "d": 0.3
    }

    wandb_config = {
        "structure":structure,
        "config":network_config,
        "database": database_report,
        "epochs": EPOCHS,
        "early_stop": EARLY_STOP,
    }
    
    wandb.init(
        project='windprognose',
        entity='leuchtum',
        notes="test",
        config=wandb_config,
    )
    
    my_model = MLForecast.networks.LSTMseq2vec(structure, network_config)
    my_model.set_input_shape(train[0][0].shape)
    my_model.set_output_shape(train[1][0].shape)
    my_model.build()
    my_model.compile()

    my_model.fit(train_np, val_np, early_stop=EARLY_STOP, epochs=EPOCHS, track_wandb=True)
    
    performance = MLForecast.postprocessing.PerformanceAnalyser(my_model.model, test_np)
    xxx = performance.by_metrics("mae")
    yyy = performance.by_metrics("mae", by_individual_column=True)
    
    #performance_report = performance.report()

    #MLForecast.visualize.boxplot(yyy, LABEL_COLUMNS, "X", "Y")
    
    print("finish")
    
    
    
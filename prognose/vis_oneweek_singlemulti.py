import tensorflow as tf
import MLForecast.networks
import MLForecast.postprocessing
import MLForecast.preprocessing as prep
import MLForecast.visualize
import pickle
import pandas as pd
import numpy as np

if __name__ == '__main__':
    PATH = "/home/daniel/Dokumente/RNNseq2vec_HORI12_HIST144_POLxy_SINGLE0/"
    INDENT = "RNNseq2vec_HORI12_HIST144_POLxy_SINGLE0"
    SEP = "_"
            
    def load_from_others(name):
        filename = PATH + INDENT + "_" + name + ".pkl"
        with open(filename, "rb") as loadfile:
            print(f"READ IN {filename}")
            f = pickle.load(loadfile)
        return f

    filename = PATH + INDENT + "_" + "dwddf" + ".pkl"
    df_dwd = pd.read_pickle(filename)
    test_np_xy = load_from_others("testnp")
    filename = PATH + INDENT + "_" + "testdf" + ".pkl"
    test_df_xy = pd.read_pickle(filename)
    
    
    PATH = "/home/daniel/Dokumente/RNNseq2vec_HORI12_HIST144_POLxy_SINGLE1/"
    INDENT = "RNNseq2vec_HORI12_HIST144_POLxy_SINGLE1"
    SEP = "_"
            
    def load_from_others(name):
        filename = PATH + INDENT + "_" + name + ".pkl"
        with open(filename, "rb") as loadfile:
            print(f"READ IN {filename}")
            f = pickle.load(loadfile)
        return f

    test_np_single = load_from_others("testnp")
    filename = PATH + INDENT + "_" + "testdf" + ".pkl"
    test_df_single = pd.read_pickle(filename)
    

    model_single = MLForecast.networks.LSTMseq2vec()
    model_xy = MLForecast.networks.LSTMseq2vec()
    model_single.load("/home/daniel/Dokumente/models/model_763e2f65.hdf5")
    model_xy.load("/home/daniel/Dokumente/models/model_2a57b566.hdf5")
    
    df_dwd = df_dwd[["WS_01886", "WD_01886"]]
    df_dwd = df_dwd.rename(columns={"WS_01886": "Windstärke", "WD_01886": "Windrichtung"})

    for n in range(1000,1150):
        root_df = df_dwd.loc[list(test_df_xy[0][n].index) + list(test_df_xy[1][n].index)]
        
        input_raw = np.expand_dims(test_df_xy[0][n].values, axis=0)
        pred_raw = model_xy.model.predict(input_raw)
        pred_df = pd.DataFrame(
            pred_raw.squeeze(),
            index=test_df_xy[1][n].index,
            columns=[key.replace("LABEL_","") for key in test_df_xy[1][n].keys()]
        )
        pred_df = prep.cart2pol(pred_df)
        pred_df_xy = pred_df.rename(columns={"WS_01886":"Vorhersage Windstärke mit allen Stationen", "WD_01886":"Vorhersage Windrichtung mit allen Stationen"})
    
        input_raw = np.expand_dims(test_df_single[0][n].values, axis=0)
        pred_raw = model_single.model.predict(input_raw)
        pred_df = pd.DataFrame(
            pred_raw.squeeze(),
            index=test_df_single[1][n].index,
            columns=[key.replace("LABEL_","") for key in test_df_single[1][n].keys()]
        )
        pred_df = prep.cart2pol(pred_df)
        pred_df_single = pred_df.rename(columns={"WS_01886":"Vorhersage Windstärke mit einer Station", "WD_01886":"Vorhersage Windrichtung mit einer Station"})
        
        plot_df = pd.concat([root_df, pred_df_single, pred_df_xy], axis=1)
        
        MLForecast.visualize.plot_label_prediction_example_both(plot_df, ["Windrichtung in °", "Windgeschwindigkeit in m/s"],save=True, n=n)
    
    print("finished")
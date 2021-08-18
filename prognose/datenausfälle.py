import pandas as pd
import pickle
from MLForecast.visualize import root_heatmap

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
    
    #root_heatmap(df_dwd, "Datenausfälle", "Zeit in h", save=True)
    root_heatmap(df_dwd.iloc[1000:1100], "Datenausfälle", "Zeit in h",index=(range(0,101,20),range(1000,1101,20)), save=True)
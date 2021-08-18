import tensorflow as tf
import MLForecast.networks
import MLForecast.postprocessing
import MLForecast.preprocessing as prep
import MLForecast.visualize
import pickle
import pandas as pd

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
    #filename = PATH + INDENT + "_" + "rootdf" + ".pkl"
    #df = pd.read_pickle(filename)

    test_np = load_from_others("testnp")
    #val_np = load_from_others("valnp")
    #train_np = load_from_others("trainnp")

    filename = PATH + INDENT + "_" + "testdf" + ".pkl"
    test_df = pd.read_pickle(filename)
    #filename = PATH + INDENT + "_" + "valdf" + ".pkl"
    #val_df = pd.read_pickle(filename)
    #filename = PATH + INDENT + "_" + "traindf" + ".pkl"
    #train_df = pd.read_pickle(filename)

    database_report = load_from_others("report")

    model = MLForecast.networks.LSTMseq2vec()
    model.load("/home/daniel/Dokumente/models/model_2a57b566.hdf5")
    
    #performance = MLForecast.postprocessing.PerformanceAnalyser(model.model, test_np)
    #xxx = performance.by_metrics("mae")
    #yyy = performance.by_metrics("mae", by_individual_column=True)
    
    #performance_report = performance.report()
    LABEL_COLUMNS = ["WX_01886", "WY_01886"]

    df2 = prep.pol2cart(df_dwd)
    
    for n in range(200,400):
        print(n)
        MLForecast.visualize.plot_label_prediction_example_cart2pol(
            model.model,
            test_df[0][n],
            df2.loc[test_df[0][n].index][LABEL_COLUMNS],
            test_df[1][n],
            None,
            ["Windrichtung in °","Windgeschwindigkeit in m/s"],
            "Windprognose",
            save=True,
            n=n)
    
    print("finished")
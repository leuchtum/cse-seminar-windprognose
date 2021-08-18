import tensorflow as tf
import MLForecast.networks
import MLForecast.postprocessing
import MLForecast.visualize
import pickle
import pandas as pd

if __name__ == '__main__':
    PATH = "/home/daniel/Dokumente/RNNseq2vec_HORI12_HIST144_POLxy_SINGLE1/"
    INDENT = "RNNseq2vec_HORI12_HIST144_POLxy_SINGLE1"
    SEP = "_"
        
    def load_from_others(name):
        filename = PATH + INDENT + "_" + name + ".pkl"
        with open(filename, "rb") as loadfile:
            print(f"READ IN {filename}")
            f = pickle.load(loadfile)
        return f

    #filename = PATH + INDENT + "_" + "dwddf" + ".pkl"
    #df_dwd = pd.read_pickle(filename)
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
    model.load("/home/daniel/Dokumente/models/model_763e2f65.hdf5")
    
    performance = MLForecast.postprocessing.PerformanceAnalyser2(model.model, test_np)
    
    yyy = performance.by_metrics("mae", by_individual_column=True)
    print("MAE Stärke")
    print(yyy[0].mean().mean())
    print("MAE Richtung")
    print(yyy[1].mean().mean())
    yyy = performance.by_metrics("mse", by_individual_column=True)
    print("MSE Stärke")
    print(yyy[0].mean().mean())
    print("MSE Richtung")
    print(yyy[1].mean().mean())
    
    #performance_report = performance.report()
    LABEL_COLUMNS = ["Windstärke", "Windrichtung"]
    #MLForecast.visualize.boxplot(yyy, "x wind", LABEL_COLUMNS, "y wind", "MAE", save=True)
    MLForecast.visualize.boxplot_ws(yyy, "Güte über Prognosehorizont", LABEL_COLUMNS, "Prognosehorizont", "MAE", save=True)
    
    print("finished")
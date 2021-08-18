import MLForecast.postprocessing
import MLForecast.networks
import pickle

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

    test_np = load_from_others("testnp")

    model = MLForecast.networks.LSTMseq2vec()
    model.load("/home/daniel/Dokumente/models/model_2a57b566.hdf5")

    performance = MLForecast.postprocessing.PerformanceAnalyser2(model.model, test_np)
    yyy = performance.by_metrics("mae", by_individual_column=True)
    print("%%%%%%% XY MULTI %%%%%%%%")
    print("MAE Stärke")
    print(yyy[0].mean().mean())
    print("MAE Richtung")
    print(yyy[1].mean().mean())
    yyy = performance.by_metrics("mse", by_individual_column=True)
    print("MSE Stärke")
    print(yyy[0].mean().mean())
    print("MSE Richtung")
    print(yyy[1].mean().mean())
    
    
    
    
    
    PATH = "/home/daniel/Dokumente/RNNseq2vec_HORI12_HIST144_POLxy_SINGLE1/"
    INDENT = "RNNseq2vec_HORI12_HIST144_POLxy_SINGLE1"
    SEP = "_"
    
    def load_from_others(name):
        filename = PATH + INDENT + "_" + name + ".pkl"
        with open(filename, "rb") as loadfile:
            print(f"READ IN {filename}")
            f = pickle.load(loadfile)
        return f

    test_np = load_from_others("testnp")

    model = MLForecast.networks.LSTMseq2vec()
    model.load("/home/daniel/Dokumente/models/model_763e2f65.hdf5")

    performance = MLForecast.postprocessing.PerformanceAnalyser2(model.model, test_np)
    yyy = performance.by_metrics("mae", by_individual_column=True)
    print("%%%%%%% XY SINGLE %%%%%%%%")
    print("MAE Stärke")
    print(yyy[0].mean().mean())
    print("MAE Richtung")
    print(yyy[1].mean().mean())
    yyy = performance.by_metrics("mse", by_individual_column=True)
    print("MSE Stärke")
    print(yyy[0].mean().mean())
    print("MSE Richtung")
    print(yyy[1].mean().mean())
    
    
    
    
    
    PATH = "/home/daniel/Dokumente/RNNseq2vec_HORI12_HIST144_POLsd_SINGLE0/"
    INDENT = "RNNseq2vec_HORI12_HIST144_POLsd_SINGLE0"
    SEP = "_"
    
    def load_from_others(name):
        filename = PATH + INDENT + "_" + name + ".pkl"
        with open(filename, "rb") as loadfile:
            print(f"READ IN {filename}")
            f = pickle.load(loadfile)
        return f

    test_np = load_from_others("testnp")

    model = MLForecast.networks.LSTMseq2vec()
    model.load("/home/daniel/Dokumente/models/model_8e5d015b.hdf5")

    performance = MLForecast.postprocessing.PerformanceAnalyser(model.model, test_np)
    yyy = performance.by_metrics("mae", by_individual_column=True)
    print("%%%%%%% SD MULTI %%%%%%%%")
    print("MAE Stärke")
    print(yyy[0].mean().mean())
    print("MAE Richtung")
    print(yyy[1].mean().mean())
    yyy = performance.by_metrics("mse", by_individual_column=True)
    print("MSE Stärke")
    print(yyy[0].mean().mean())
    print("MSE Richtung")
    print(yyy[1].mean().mean())
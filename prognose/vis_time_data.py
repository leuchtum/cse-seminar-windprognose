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

    filename = PATH + INDENT + "_" + "rootdf" + ".pkl"
    df = pd.read_pickle(filename)
        
    MLForecast.visualize.plotsin(df["CALENDAR_COS_HOUR"].head(25), save="vis_sin_day", ticks=(range(0,25,3),["0:00","3:00","6:00","9:00","12:00","15:00","18:00","21:00","0:00"]))
    MLForecast.visualize.plotsin(df["CALENDAR_COS_SEAS"].head(8761), save="vis_sin_year")
    print("finished")
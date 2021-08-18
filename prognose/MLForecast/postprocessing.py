import tensorflow as tf
import pandas as pd
import numpy as np
from .preprocessing import cart2pol
class PerformanceAnalyser:
    def __init__(self, model, test):
        self.model = model
        self.test = test
        self.label = test[1]
        self.pred = model.predict(test[0])
    
    def by_metrics(self, metric, by_individual_column=False):
        if by_individual_column:
            convert =lambda a: [a[:,:,[i]] for i in range(a.shape[-1])]
            label = convert(self.label)
            pred = convert(self.pred)
        else:
            label = [self.label]
            pred = [self.pred]
            
        dfs = []
        for l, p in zip(label, pred):
            if metric == "mae":
                vals = tf.keras.metrics.MAE(l, p).numpy()
            elif metric == "mse":
                vals = tf.keras.metrics.MSE(l, p).numpy()
            
            col_names = [f"t+{i+1}" for i in range(vals.shape[-1])]
            dfs.append(pd.DataFrame(vals, columns=col_names))
        
        return dfs[0] if len(dfs) == 1 else dfs
    

    def dif(self):
        convert =lambda a: [a[:,:,i] for i in range(a.shape[-1])]
        label = convert(self.label)
        pred = convert(self.pred)
        
        dfs=[]
        for l, p in zip(label, pred):
            vals = l - p
            col_names = [f"t+{i+1}" for i in range(vals.shape[-1])]
            dfs.append(pd.DataFrame(vals, columns=col_names))
            
        return dfs[0] if len(dfs) == 1 else dfs
    
    def report(self):
        return {
            "mae": self.by_metrics("mae").mean().mean(),
            "mse": self.by_metrics("mse").mean().mean(),
        }
        
        
class PerformanceAnalyser2:
    def __init__(self, model, test):
        self.model = model
        self.test = test
        label_dfs = [pd.DataFrame(a, columns=["WX","WY"]) for a in test[1]]
        label_dfs = [cart2pol(df) for df in label_dfs]
        self.label = np.array([df.to_numpy() for df in label_dfs])
        
        pred = model.predict(test[0])
        pred_dfs = [pd.DataFrame(a, columns=["WX","WY"]) for a in pred]
        pred_dfs = [cart2pol(df) for df in pred_dfs]
        self.pred = np.array([df.to_numpy() for df in pred_dfs])
        
    
    def by_metrics(self, metric, by_individual_column=False):
        if by_individual_column:
            convert =lambda a: [a[:,:,[i]] for i in range(a.shape[-1])]
            label = convert(self.label)
            pred = convert(self.pred)
        else:
            label = [self.label]
            pred = [self.pred]
            
        dfs = []
        for l, p in zip(label, pred):
            if metric == "mae":
                vals = tf.keras.metrics.MAE(l, p).numpy()
            elif metric == "mse":
                vals = tf.keras.metrics.MSE(l, p).numpy()
            
            col_names = [f"t+{i+1}" for i in range(vals.shape[-1])]
            dfs.append(pd.DataFrame(vals, columns=col_names))
        
        return dfs[0] if len(dfs) == 1 else dfs

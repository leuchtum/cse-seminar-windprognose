import tensorflow as tf
import pandas as pd

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
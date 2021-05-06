import numpy as np

class MinMaxNormalizer:
    def __init__(self, default_border=(0,1), spezial_border={}, exclude=[]) -> None:
        self.default_border = default_border
        self.spezial_border = spezial_border
        self.exclude = exclude
        self.parmas = {}
        self.borders = {}

    def fit(self, df):
        
        for colname in df:
            skip = any(ex in colname for ex in self.exclude)

            if not skip:
                border = None
                for spezial in self.spezial_border:
                    if spezial in colname:
                        border = self.spezial_border[spezial]
                        break
                if not border:
                    border = self.default_border
                    
                self.parmas[colname] = (df[colname].min(), df[colname].max())
                self.borders[colname] = border

    def normalize(self, df):
        df = df.copy()

        for colname in df:
            if colname in self.parmas:
                params = self.parmas[colname]
                borders = self.borders[colname]
                df[colname]=df[colname].apply(np.interp, args=(params, borders))
            
        return df

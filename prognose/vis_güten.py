import MLForecast.visualize
import pandas as pd

if __name__ == '__main__':
    mae_s = {
        "XY-Vektorisierung\nAlle Stationen": 0.8485,
        "XY-Vektorisierung\nEine Station": 0.9966,
        "Ohne XY-Vektorisierung\nAlle Stationen": 1.0598
        }

    mae_d = {
        "XY-Vektorisierung\nAlle Stationen": 61.1353,
        "XY-Vektorisierung\nEine Station": 62.5814,
        "Ohne XY-Vektorisierung\nAlle Stationen": 63.8975
        }

    mse_s = {
        "XY-Vektorisierung\nAlle Stationen": 1.3274,
        "XY-Vektorisierung\nEine Station": 1.9019,
        "Ohne XY-Vektorisierung\nAlle Stationen": 1.7545
        }

    mse_d = {
        "XY-Vektorisierung\nAlle Stationen": 10581,
        "XY-Vektorisierung\nEine Station": 10251,
        "Ohne XY-Vektorisierung\nAlle Stationen": 7075
        }

    MLForecast.visualize.barplot(pd.Series(mae_s), pd.Series(mae_d), "MAE", save="mae")
    MLForecast.visualize.barplot(pd.Series(mse_s), pd.Series(mse_d), "MSE", save="mse")
    
    print("finished")
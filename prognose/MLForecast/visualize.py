import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_input_label_example(input_df, label_df):
    fig, axes = plt.subplots(2, 1, sharex=True)
    
    sns.lineplot(ax=axes[0], data=input_df, markers=True, dashes=False)
    axes[0].set_title("INPUT")
    axes[0].grid()
    
    sns.lineplot(ax=axes[1], data=label_df, markers=True, dashes=False)
    axes[1].set_title("LABEL")
    axes[1].grid()
    
    plt.show()
    
def plot_input_label_prediction_example(model, input_df, label_df):
    fig, axes = plt.subplots(2, 1, sharex=True)
    
    sns.lineplot(ax=axes[0], data=input_df, markers=True, dashes=False)
    axes[0].set_title("INPUT")
    axes[0].grid()
    
    input_raw = np.expand_dims(input_df.values, axis=0)
    pred_raw = model.predict(input_raw)
    pred_df = pd.DataFrame(
        pred_raw.squeeze(),
        index=label_df.index,
        columns=[key.replace("LABEL", "PREDICTION") for key in label_df.keys()]
    )
    plot_df = pd.concat([label_df, pred_df], axis=1)
    sns.lineplot(ax=axes[1], data=plot_df, markers=True, dashes=False)
    axes[1].set_title("LABEL + PREDICTION")
    axes[1].grid()
    
    plt.show()  
    
def boxplot(dfs, subtitles, xtitle, ytitle):
    assert type(dfs) == list
    
    if len(dfs) != 1:
        fig, axes = plt.subplots(1, len(dfs), sharey=True)
        axes[0].set_xlabel(xtitle)
        for i in range(len(dfs)):
            axes[i].grid(axis="y", which="major")
            axes[i].set_axisbelow(True)
            sns.boxplot(ax=axes[i], data=dfs[i])
            axes[i].set_title(subtitles[i])
            axes[i].set_ylabel(ytitle)
            
    else:
        plt.grid(axis="y", which="major")
        plt.set_axisbelow(True)
        plt.xlabel(xtitle)
        sns.boxplot(data=dfs[0])
        plt.title(subtitles[0])
        plt.ylabel(ytitle)
        
    
    plt.show()
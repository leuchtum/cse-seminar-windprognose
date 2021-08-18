import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
from .preprocessing import cart2pol

sns.set_style("whitegrid")
sns.set_context("notebook")
COLOR = mcolors.CSS4_COLORS["lightsteelblue"]
FIGSIZE = (14,7)
DPI = 400


def plot_label_prediction_example_both(df, ytitle, save=False, n=None):
    fig, axes = plt.subplots(2,1 , sharex=True, figsize=FIGSIZE)
    
    plotnow = df[["Windrichtung","Vorhersage Windrichtung mit allen Stationen","Vorhersage Windrichtung mit einer Station"]]
    sns.lineplot(ax = axes[0], data=plotnow, dashes=False, linewidth = 3)
    
    plotnow = df[["Windstärke","Vorhersage Windstärke mit allen Stationen","Vorhersage Windstärke mit einer Station"]]
    sns.lineplot(ax = axes[1], data=plotnow, dashes=False, linewidth = 3)

    idx = df.index
    axes[0].plot([idx[-12],idx[-12]], [-10,370], "k:")
    axes[1].plot([idx[-12],idx[-12]], [-1,16], "k:")
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    
    axes[0].set_ylim(0,360)
    axes[1].set_ylim(0,14)
    
    axes[0].set_ylabel(ytitle[0])
    axes[1].set_ylabel(ytitle[1])
    
    if save:
        if n:
            savefig(f"both_{n}")
        else:
            savefig("both")
    else:
        plt.show()
        
def plot_label_prediction_example_ws(model, model_input_df, input_df, label_df, xtitle, ytitle, title, save=False, n=None):
    fig, axes = plt.subplots(2,1 , sharex=True, figsize=FIGSIZE)
    
    input_df = input_df.rename(columns={"WS_01886":"Windstärke", "WD_01886":"Windrichtung"})
    
    label2_df = label_df.rename(columns={"LABEL_WS_01886":"WX_01886", "LABEL_WD_01886":"WY_01886"})
    
    label2_df = label2_df.rename(columns={"WX_01886":"Windstärke", "WY_01886":"Windrichtung"})
    
    plot_df = pd.concat([input_df, label2_df])
    
    input_raw = np.expand_dims(model_input_df.values, axis=0)
    pred_raw = model.predict(input_raw)
    pred_df = pd.DataFrame(
        pred_raw.squeeze(),
        index=label_df.index,
        columns=[key.replace("LABEL_","") for key in label_df.keys()]
    )
    
    pred_df = pred_df.rename(columns={"WS_01886":"Vorhersage Windstärke", "WD_01886":"Vorhersage Windrichtung"})
    
    plotnow = pd.concat([plot_df["Windrichtung"],pred_df["Vorhersage Windrichtung"]], axis=1)
    sns.lineplot(ax = axes[0], data=plotnow, dashes=False, linewidth = 3)
    
    plotnow = pd.concat([plot_df["Windstärke"],pred_df["Vorhersage Windstärke"]], axis=1)
    sns.lineplot(ax = axes[1], data=plotnow, dashes=False, linewidth = 3)

    idx = input_df.index
    axes[0].plot([idx[-1],idx[-1]], [-10,370], "k:")
    axes[1].plot([idx[-1],idx[-1]], [-1,16], "k:")
    #plt.plot([idx[-1],idx[-1]], [-10,10], "k:")
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    
    axes[0].set_ylim(0,360)
    axes[1].set_ylim(0,14)
    
    axes[0].set_ylabel(ytitle[0])
    axes[1].set_ylabel(ytitle[1])
    
    if save:
        if n:
            savefig(f"{title}_{n}")
        else:
            savefig(title)
    else:
        plt.show()
        
        
def plotsin(series, save=None, ticks=None):
    fig = plt.figure(figsize=(6,2.5))
    
    if ticks:
        series.index = range(len(series))
        plt.plot(series)
        plt.xticks(*ticks)
    else:
        plt.plot(series)    
    
    if save:
        savefig(save)
    else:
        plt.show()
    
def root_heatmap(df, title, xlabel, index=None, save=False, rechteck = False):
    fig = plt.figure(figsize=FIGSIZE)
    #Wie index?: z.B. (range(0,501,100),range(6000,6501,100))
    plt.pcolor((df.isna()*1).T, cmap = "RdYlGn_r")
    
    plt.yticks(np.arange(0.5, len(df.columns), 1), df.columns)
    if index:
        plt.xticks(*index)
    if rechteck:
        plt.plot([rechteck[0],rechteck[0]],[0,len(df.columns)],"k",linewidth=2)
        plt.plot([rechteck[1],rechteck[1]],[0,len(df.columns)],"k",linewidth=2)
        
    plt.title(title)
    plt.xlabel(xlabel)
    
    if save:
        savefig(title)
    else:
        plt.show()


def footnote(axes, note):
    axes.annotate(
        note, 
        xy = (1.0, -0.1),
        xycoords='axes fraction',
        ha='right',
        va="center", 
        fontsize=8
    )
    
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
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE)
    
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
    
    
def plot_label_prediction_example(model, model_input_df, input_df, label_df, xtitle, ytitle, title, save=False, n=None):
    fig = plt.figure(figsize=FIGSIZE)
    
    input_df = input_df.rename(columns={"WX_01886":"x-Windkomponente", "WY_01886":"y-Windkomponente"})
    label_df = label_df.rename(columns={"LABEL_WX_01886":"x-Windkomponente", "LABEL_WY_01886":"y-Windkomponente"})
    plot_df = pd.concat([input_df, label_df])
    
    input_raw = np.expand_dims(model_input_df.values, axis=0)
    pred_raw = model.predict(input_raw)
    pred_df = pd.DataFrame(
        pred_raw.squeeze(),
        index=label_df.index,
        columns=[key + " Vorhersage" for key in label_df.keys()]
    )
    
    sns.lineplot(data=pd.concat([plot_df, pred_df], axis=1), dashes=False, linewidth = 3)
    idx = input_df.index
    plt.plot([idx[-1],idx[-1]], [-10,10], "k:")
    ax = plt.gca()
    ax.legend(loc="upper left")
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)
    plt.ylim((-6, 6))
    
    if save:
        if n:
            savefig(f"{title}_{n}")
        else:
            savefig(title)
    else:
        plt.show()



def plot_label_prediction_example_cart2pol(model, model_input_df, input_df, label_df, xtitle, ytitle, title, save=False, n=None):
    fig, axes = plt.subplots(2,1 , sharex=True, figsize=FIGSIZE)
    
    input_df = cart2pol(input_df)
    input_df = input_df.rename(columns={"WS_01886":"Windstärke", "WD_01886":"Windrichtung"})
    
    label2_df = label_df.rename(columns={"LABEL_WX_01886":"WX_01886", "LABEL_WY_01886":"WY_01886"})
    label2_df = cart2pol(label2_df)
    label2_df = label2_df.rename(columns={"WS_01886":"Windstärke", "WD_01886":"Windrichtung"})
    
    plot_df = pd.concat([input_df, label2_df])
    
    input_raw = np.expand_dims(model_input_df.values, axis=0)
    pred_raw = model.predict(input_raw)
    pred_df = pd.DataFrame(
        pred_raw.squeeze(),
        index=label_df.index,
        columns=[key.replace("LABEL_","") for key in label_df.keys()]
    )
    pred_df = cart2pol(pred_df)
    pred_df = pred_df.rename(columns={"WS_01886":"Vorhersage Windstärke", "WD_01886":"Vorhersage Windrichtung"})
    
    plotnow = pd.concat([plot_df["Windrichtung"],pred_df["Vorhersage Windrichtung"]], axis=1)
    sns.lineplot(ax = axes[0], data=plotnow, dashes=False, linewidth = 3)
    
    plotnow = pd.concat([plot_df["Windstärke"],pred_df["Vorhersage Windstärke"]], axis=1)
    sns.lineplot(ax = axes[1], data=plotnow, dashes=False, linewidth = 3)

    idx = input_df.index
    axes[0].plot([idx[-1],idx[-1]], [-10,370], "k:")
    axes[1].plot([idx[-1],idx[-1]], [-1,16], "k:")
    #plt.plot([idx[-1],idx[-1]], [-10,10], "k:")
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    
    axes[0].set_ylim(0,360)
    axes[1].set_ylim(0,14)
    
    axes[0].set_ylabel(ytitle[0])
    axes[1].set_ylabel(ytitle[1])
    
    if save:
        if n:
            savefig(f"{title}_{n}")
        else:
            savefig(title)
    else:
        plt.show()
        
        
def boxplot(dfs, suptitle, subtitles, xtitle, ytitle, note=None, save=False):
    assert type(dfs) == list
    
    if len(dfs) != 1:
        fig, axes = plt.subplots(1, len(dfs), sharey=True, figsize=FIGSIZE)#, dpi=DPI)
        axes[0].set_ylabel(ytitle)
        fig.suptitle(suptitle)
        for i in range(len(dfs)):
            sns.boxplot(ax=axes[i], data=dfs[i], color=COLOR)
            axes[i].set_title(subtitles[i])
            axes[i].set_xlabel(xtitle)
        if note:
            footnote(axes[-1], note)
            
    else:
        raise("IMPLEMENT: dpi, figsize and footnote")
        sns.boxplot(data=dfs[0])
        plt.xlabel(xtitle)
        plt.title(subtitles[0])
        plt.ylabel(ytitle)
        
    plt.margins(.05,.95)
    
    if save:
        savefig(suptitle)
    else:
        plt.show()

def boxplot_ws(dfs, suptitle, subtitles, xtitle, ytitle, note=None, save=False):
    assert type(dfs) == list
    
    fig, axes = plt.subplots(1, len(dfs), figsize=FIGSIZE)#, dpi=DPI)
    axes[0].set_ylabel(ytitle)
    fig.suptitle(suptitle)
    for i in range(len(dfs)):
        sns.boxplot(ax=axes[i], data=dfs[i], color=COLOR)
        axes[i].set_title(subtitles[i])
        axes[i].set_xlabel(xtitle)
    if note:
        footnote(axes[-1], note)

    axes[0].set_ylim(-.1,10)
    axes[1].set_ylim(-1,360)
    plt.margins(.05,.95)
    
    if save:
        savefig(suptitle)
    else:
        plt.show()

def barplot(left_series, right_series, ylabel, save=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    
    sns.barplot(x=left_series.index, y=left_series.values, palette="rocket", ax=ax1)
    sns.barplot(x=right_series.index, y=right_series.values, palette="rocket", ax=ax2)
    
    ax1.set_ylabel(ylabel)
    ax2.set_ylabel(ylabel)
    
    ax1.set_title("Windgeschwindigkeit")
    ax2.set_title("Windrichtung")
    
    if save:
        savefig(f"barplot_{save}")
    else:
        plt.show()
        
        
def savefig(name):
    plt.savefig(f'/home/daniel/Bilder/{name}.png')#, bbox_inches='tight', dpi=DPI)
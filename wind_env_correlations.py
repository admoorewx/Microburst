import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


csv_path = "E:/Microburst/sounding_environments.csv"
save_dir = "E:/Microburst/images/analysis/"


def plot_scatter(x,y,title,xlabel,ylabel,save_dir,save_name):
    corr = np.corrcoef(x,y)[0][1]
    print(f'{xlabel} : {ylabel} correlation: {corr}')
    plt.figure()
    plt.scatter(x,y,color='k',label=f'Correlation: {round(corr,2)}')
    plt.axhline(y=26.0,color='k',linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(save_dir,save_name))

def scatter_3d(x,y,z,title,xlabel,ylabel,save_dir,save_name):
    plt.figure()
    scatt = plt.scatter(x,y,c=z,cmap='RdYlBu_r',edgecolors='k',vmin=15.0,vmax=26.0,alpha=0.75)
    plt.colorbar(scatt)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(save_dir,save_name))
    plt.close()

# Read in the csv
df = pd.read_csv(csv_path)
windspeeds = df.iloc[:,1].values
ylabel = "Max Surface Wind (m/s)"
combinations = itertools.combinations(df.columns.values[2:],2)
for combo in combinations:
    xlabel= combo[0]
    ylabel = combo[1]
    xloc = list(df.columns.values).index(combo[0])
    yloc = list(df.columns.values).index(combo[1])
    x = df.iloc[:,xloc]
    y = df.iloc[:,yloc]
    title = f'{xlabel} + {ylabel} vs. Max Winds'
    save_name = f'{xlabel}_{ylabel}_max_wind.png'
    scatter_3d(x,y,windspeeds,title,xlabel,ylabel,save_dir,save_name)


# for i in range(2,len(df.columns)):
#     variable = df.iloc[:,i].values
#     xlabel = df.columns.values[i]
#     title = f'{xlabel} vs. Max Wind'
#     save_name = f'{xlabel}_max_wind.png'
#     plot_scatter(variable,windspeeds,title,xlabel,ylabel,save_dir,save_name)
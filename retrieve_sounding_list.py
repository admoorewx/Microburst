import os
import pandas as pd
import shutil

sounding_list_csv = "/scratch/admoore/valid_soundings.csv"
list_directory = "/home/admoore/CM1/soundings/grid_soundings/snd/"
target_directory = "/scratch/admoore/valid_soundings/"

sounding_list = pd.read_csv(sounding_list_csv)
sounding_list = sounding_list.values[:,0]
for filename in os.listdir(list_directory):
    if filename in sounding_list:
        shutil.copyfile(os.path.join(list_directory,filename),os.path.join(target_directory,filename))

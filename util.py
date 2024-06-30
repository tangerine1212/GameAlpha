import numpy as np 
import os
import cv2
import sys
import time
import pandas as pd
import pickle
from ray.tune.registry import register_trainable

def get_winrate_and_weight(logdir,league):
    wr_path = os.path.join(logdir,'winrates.csv')
    weight_path = os.path.join(logdir,'weights')
    
    wr = pd.read_csv(wr_path)
    winrates = wr.values[0][1:]
    
    weights = os.listdir(weight_path)
    weights = [i for i in weights if i.split('.')[-1] == 'pkl']
    
    minlen = min(len(weights),len(winrates))
    
    winrates = winrates[-minlen:]
    weights = weights[-minlen:]
    
    weights = sorted(weights,key=lambda x:int(x.split('.')[0].split("_")[-1]))
    
    assert(len(weights) == len(winrates))
    
    weights = [pickle.load(open(os.path.join(weight_path,i), "rb")) for i in weights]
    
    for weight in weights:
        league.add_weight.remote(weight)
    league.set_winrates.remote(winrates)
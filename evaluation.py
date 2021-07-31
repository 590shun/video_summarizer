import numpy as np
import h5py
import pandas as pd
import statistics
import matplotlib
from matplotlib import pyplot as plt
from utils.eval import evaluate_scores

# Evaluation (Kendall's & Spearman's correlation)
# setting
m = h5py.File('./logs/timestamp_TransformerTrainer/dataset_splits.json_preds.h5', 'r')


u = h5py.File('./datasets/summarizer_dataset.h5', 'r')
# load video list
path_text = './video_list.txt'
with open(path_text) as f:
    video_list = f.read().splitlines()

i = 1
kendall_list = []
spearman_list = []

for video in video_list:
    usr = u['video_' + str(i)]['user_scores'][()]
    machine = m['summarizer_dataset_as.h5']['video_' + str(i)]['machine_scores'][()]
    spe = evaluate_scores(machine, usr, metric="spearmanr")
    ken = evaluate_scores(machine, usr, metric="kendalltau")
    spearman_list.append(spe)
    kendall_list.append(ken)

kendall = statistics.mean(kendall_list)
spearman = statistics.mean(spearman_list)

print('kendallの相関係数:', kendall)
print('spearmanの相関係数:', spearman)

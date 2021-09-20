import numpy as np
import h5py
import pandas as pd
import statistics
import matplotlib
from matplotlib import pyplot as plt
from utils.eval import evaluate_scores

# Evaluation (Kendall's & Spearman's correlation)
# setting
m = h5py.File('./logs/1632145246_DSNTrainer/summe_splits.json_preds.h5', 'r')
u = h5py.File('./datasets/summarizer_dataset_summe_google_pool5.h5', 'r')

# load video list
path_text = './summe_video_list.txt'
with open(path_text) as f:
    video_list = f.read().splitlines()

kendall_list = []
spearman_list = []

for i in range(len(video_list)):
    i += 1
    usr = u['video_' + str(i)]['user_scores'][()]
    machine = m['summarizer_dataset_summe_google_pool5.h5']['video_' + str(i)]['machine_scores'][()]
    spe = evaluate_scores(machine, usr, metric="spearmanr")
    ken = evaluate_scores(machine, usr, metric="kendalltau")
    spearman_list.append(spe)
    kendall_list.append(ken)

kendall = statistics.mean(kendall_list)
spearman = statistics.mean(spearman_list)

print('kendall:', kendall)
print('spearman:', spearman)

# -*- coding:utf-8 -*-

### 抽出した動画特徴量を合成してhdf形式のデータセットを作成するスクリプト ###
import numpy as np
import h5py
import pandas as pd
import matplotlib
from KTS.cpd_nonlin import cpd_nonlin
from KTS.cpd_auto import cpd_auto
from matplotlib import pyplot as plt

# Kernel Temporal Segmentationに使用
def gen_data(n, m, d=1024):
    """
    変化位置を生成する関数
    ●入力
    ・n: 動画のフレーム数
    ・m: 変化位置の数
    ・d: 次元数(今回は1024)
    ●返り値
    ・numpy: (n X d)
    ・cps: 切り替わり箇所が格納された配列
    """
    # 乱数固定の乱数配列を生成
    np.random.seed(1)
    # Select changes at some distance from the boundaries
    cps = np.random.permutation((n*3//4)-1)[0:m] + 1 + n//8
    cps = np.sort(cps)
    cps = [0] + list(cps) + [n]
    mus = np.random.rand(m+1, d)*(m/2)  # make sigma = m/2
    X = np.zeros((n, d))
    for k in range(m+1):
        X[cps[k]:cps[k+1], :] = mus[k, :][np.newaxis, :] + np.random.rand(cps[k+1]-cps[k], d)
    return (X, np.array(cps))

# 動画のリストを読み込む
path_text = './video_list.txt'
with open(path_text) as f:
    video_list = f.read().splitlines()
h5_file_name = './datasets/summarizer_hoge_dataset.h5'
f = h5py.File(h5_file_name, 'a')

i = 1

for video in video_list:
    # 動画の名前を格納
    # f.create_dataset(video + '/video_name', data=data_of_name)
    # featureの設定
    feature_path = './features/' + str(video) + '_mstcn.npy'
    feature = np.load(feature_path).transpose()
    # print(video)
    # print(feature.shape)
    f.create_dataset('video_' + str(i) + '/features', data=feature)

    # gtscoresの設定
    csv_path = './gtscores/' + str(video) + '.csv'
    csv_file = pd.read_csv(csv_path, header=None)
    ann_list = list(csv_file[1])
    length = len(ann_list)
    ann_array = np.array(ann_list, dtype=float)
    f.create_dataset('video_' + str(i) + '/gtscore', data=ann_array)

    # user_scoresの設定
    csv_scr_path = './user_scores/' + str(video) + '.csv'
    csv_scr_file = pd.read_csv(csv_scr_path, header=None)
    usr_scr_list = list(csv_scr_file[1])
    scr_array = np.array(usr_scr_list, dtype=float).reshape(1, len(ann_list))
    f.create_dataset('video_' + str(i) + '/user_scores', data=scr_array)

    # n_framesの設定
    n_frames = len(ann_list)
    f.create_dataset('video_' + str(i) + '/n_frames', data=n_frames)

    # n_stepsの設定
    n_steps = len(ann_list)
    f.create_dataset('video_' + str(i) + '/n_steps', data=n_steps)

    # picksの設定
    picks = list(range(len(ann_list)))
    f.create_dataset('video_' + str(i) + '/picks', data=picks)

    # change_pointsの設定
    n, d = feature.shape
    n = n-1
    m = 20
    plt.figure("Test: automatic selection of the number of change-points")
    (X, cps_gt) = gen_data(n, m)
    print( "Ground truth: (m=%d)" % m, cps_gt)
    plt.plot(X)
    K = np.dot(X, X.T)
    #cps, scores = cpd_auto(K, 2*m, 1)
    cps, scores = cpd_auto(K, m, 1)
    print( "Estimated: (m=%d)" % len(cps), cps)
    cps_lst = cps.tolist()
    mi = np.min(X)
    ma = np.max(X)
    for cp in cps:
        plt.plot([cp, cp], [mi, ma], 'r')
    # plt.show()
    plt.savefig('./KTS/figure/' + video + '.jpg')
    
    for idx in range(len(cps_lst)):
        if idx == 0:
            array = np.array([0, cps_lst[idx]]).reshape(1,2)
        elif idx == len(cps_lst) - 1:
            array_l = np.array([cps_lst[idx-1]+1, length -1]).reshape(1,2)
            array = np.concatenate([array, array_l])
        else:
            array_m = np.array([cps_lst[idx-1]+1, cps_lst[idx]]).reshape(1,2)
            array = np.concatenate([array, array_m])
    f.create_dataset('video_' + str(i) + '/change_points', data=array)

    # n_frame_per_segの設定
    nps = []
    seg_list = array.tolist()

    for id in range(len(seg_list)):
        cnt_fr = array[id][1] - array[id][0] + 1
        nps.append(cnt_fr)
    print('n_frames_per_seg:', nps)
    nps_array = np.array(nps)
    f.create_dataset('video_' + str(i) + '/n_frame_per_seg', data=nps_array)

    # gt_summaryの設定
    gts = []
    path_txt = './gtsummary/' + video + '.txt'
    with open(path_txt) as h:
        lines = h.read().splitlines()
    for t in range(len(lines)):
        if lines[t] == 'background':
            gts.append(float(0))
        elif lines[t] == 'summary_seg':
            gts.append(float(1))
    
    gts_array = np.array(gts)
    print(gts_array.shape)
    print(type(gts_array))
    f.create_dataset('video_' + str(i) + '/gtsummary', data=gts_array)

    # user_summaryの設定
    usrs = []
    path_txt = './gtsummary/' + video + '.txt'
    with open(path_txt) as g:
        lines = g.read().splitlines()
    
    for t2 in range(len(lines)):
        if lines[t2] == 'background':
            usrs.append(float(0))
        elif lines[t2] == 'summary_seg':
            usrs.append(float(1))
    
    usrs_array = np.array(usrs).reshape(1,len(lines))
    print(usrs_array.shape)
    f.create_dataset('video_' + str(i) + '/user_summary', data=usrs_array)

    print('*'*80)

    i += 1

f.close()

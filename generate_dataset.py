# -*- coding:utf-8 -*-

### 抽出した動画特徴量を合成して.h5に格納するスクリプト ###
import numpy as np
import h5py
import pandas as pd
import matplotlib
from KTS.cpd_nonlin import cpd_nonlin
from KTS.cpd_auto import cpd_auto
from matplotlib import pyplot as plt

# Kernel temporal Segmentationに使用
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
annotator_path = './annotator_list.txt'
with open(path_text) as f:
    video_list = f.read().splitlines()
with open(annotator_path) as ann:
    annotator_list = ann.read().splitlines()
h5_file_name = './datasets/summarizer_panasonic_dataset_kts.h5'
f = h5py.File(h5_file_name, 'a')

i = 1

for video in video_list:
    # 動画の名前を格納
    # f.create_dataset(video + '/video_name', data=data_of_name)
    # featureの設定
    feature_path = './features/' + str(video) + '_mstcn.npy'
    feature = np.load(feature_path).transpose()
    f.create_dataset('video_' + str(i) + '/features', data=feature)

    # gtscoresの設定
    gtscore_path = './gtscores/' + str(video) + '.txt'
    with open(gtscore_path) as gs:
        gtscore_list = gs.read().splitlines()
    gtscore_array = np.array(gtscore_list, dtype=float)
    f.create_dataset('video_' + str(i) + '/gtscore', data=gtscore_array)

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

    for an in range(len(annotator_list)):
        # user_scoresの設定
        txt_scr_path = './user_scores/' + str(annotator_list[an]) + '/' + str(video) + '.txt'
        with open(txt_scr_path, "r") as b:
            usr_scr_list = b.read().splitlines()

        for it in range(len(usr_scr_list)):
            usr_scr_list[it] = int(usr_scr_list[it])

        if an == 0:
            usr_scr_array = np.array(usr_scr_list, dtype=float).reshape(1, len(usr_scr_list))
        else:
            usr_scr_array_2 = np.array(usr_scr_list, dtype=float).reshape(1, len(usr_scr_list))
            usr_scr_array = np.concatenate([usr_scr_array, usr_scr_array_2])
        
        # user_summaryの設定
        txt_sum_path = './user_summary/' + str(annotator_list[an]) + '/' + str(video) + '.txt'
        usrs = []
        with open(txt_sum_path) as sm:
            usr_sum_list = sm.read().splitlines()

        for s in range(len(usr_sum_list)):
            if usr_sum_list[s] == 'background':
                usrs.append(float(0))
            elif usr_sum_list[s] == 'summary_seg':
                usrs.append(float(1))
            else:
                print('error!', s)
        if an == 0:
            usr_sum_array = np.array(usrs, dtype=float).reshape(1, len(usr_sum_list))
        else:
            usr_sum_array_2 = np.array(usrs, dtype=float).reshape(1, len(usr_sum_list))
            usr_sum_array = np.concatenate([usr_sum_array, usr_sum_array_2]) 
    f.create_dataset('video_' + str(i) + '/user_scores', data=usr_scr_array)
    f.create_dataset('video_' + str(i) + '/user_summary', data=usr_sum_array)

    # n_framesの設定
    n_frames = len(usr_scr_list)
    f.create_dataset('video_' + str(i) + '/n_frames', data=n_frames)

    # n_stepsの設定
    n_steps = len(usr_scr_list)
    f.create_dataset('video_' + str(i) + '/n_steps', data=n_steps)

    # picksの設定
    picks = list(range(len(usr_scr_list)))
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
    cps, scores = cpd_auto(K, m, 1)
    print( "Estimated: (m=%d)" % len(cps), cps)
    cps_lst = cps.tolist()
    mi = np.min(X)
    ma = np.max(X)
    for cp in cps:
        plt.plot([cp, cp], [mi, ma], 'r')
    plt.savefig('./KTS/figure/' + video + '.jpg')
    
    for idx in range(len(cps_lst)):
        if idx == 0:
            array = np.array([0, cps_lst[idx]]).reshape(1,2)
        elif idx == len(cps_lst) - 1:
            array_l = np.array([cps_lst[idx-1]+1, n_steps -1]).reshape(1,2)
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

    print('*'*80)

    i += 1

f.close()

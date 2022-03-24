from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from os import listdir
import datetime as time
from utilities import Batch
from ISPForest import StreamingHalfSpaceTree

def run_report(data, n_trees, max_depth, adaptive):
    X = data[:, :-1]
    y = data[:, -1]
    batch = Batch(X, y)
    window_size = 1000
    train_size = window_size
    starttime = time.datetime.now()
    hst = StreamingHalfSpaceTree(window_size=window_size,
                                n_trees=n_trees,
                                max_depth=max_depth,
                                adaptive=adaptive)
    train_data, _ = batch.next(train_size)
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    hst.fit(train_data)
    test_res = []
    test_consistency = []
    ground_truth = []
    n_feedback_normal, n_feedback_abnormal = 0, 0
    feed_back_per_window = []
    feed_back_in_window = 0
    n_windows = 0
    while batch.epochs <= 1:
        test_data, test_label = batch.next(window_size)
        test_data = scaler.transform(test_data)
        n_windows += 1
        feed_back_in_window = 0
        for atd, label in zip(test_data, test_label):
            ground_truth.append(int(label))
            predy, consistency = hst.predict(atd, cut=True)
            if ((np.random.rand())<predy):
                hst.feed_back(atd, int(label))
                feed_back_in_window += 1
                if int(label) == 1:
                    n_feedback_abnormal += 1
                else:
                    n_feedback_normal += 1
            test_res.append(predy)
            test_consistency.append(consistency)
        feed_back_per_window.append(feed_back_in_window)
    endtime = time.datetime.now()
    runtime = (endtime - starttime).seconds * 1000
    print(ground_truth)
    print(test_res)
    print(feed_back_per_window)
    # print(n_feedback_normal, n_feedback_abnormal)
    print(roc_auc_score(ground_truth, test_res))
    return roc_auc_score(ground_truth, test_res), f1_score(ground_truth, test_res), runtime, feed_back_per_window


def HST(data):
    X = data[:, :-1]
    y = data[:, -1]
    batch = Batch(X, y)
    window_size = 250
    train_size = window_size
    starttime = time.datetime.now()
    hst = StreamingHalfSpaceTree(window_size=window_size,
                                n_trees=10,
                                max_depth=10,#改变maxdepth
                                adaptive=0.9)#改变adaptive

    train_data, _ = batch.next(train_size)
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    hst.fit(train_data)
    test_res = []
    ground_truth = []
    while batch.epochs <= 1:
        test_data, test_label = batch.next(window_size)
        # test_data, test_label = batch._reinit(window_size)
        test_data = scaler.transform(test_data)
        for atd, label in zip(test_data, test_label):
            ground_truth.append(int(label))
            test_res.append(hst.predict(atd, cut=False, scale_score=True))
    endtime = time.datetime.now()
    runtime = (endtime - starttime).seconds * 1000
    print(runtime)

if __name__ == "__main__":
    datadir = 'D:/PyCharm 2016.3.2/untitled/data/'
    # filelist = [af for af in listdir(datadir) if af.endswith('.csv')]
    filelist = ['mulcross.csv']
    for af in filelist:
        n_trees = [5, 10, 15, 20, 25, 30]
        depth = [7]#[5, 7, 9, 10, 12]
        adaptive = [0.4]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        datecontent = pd.read_csv(datadir+af).values
        fp = open(datadir+af+'.windowsize.txt', 'a')
        for n in n_trees:
            for d in depth:
                for a in adaptive:
                    auc, fmeasure, runtime, feedback = run_report(datecontent, n, d, a)
                    # print(af, auc)
                    res = np.vstack([auc, fmeasure, runtime]).T
                    feed = np.vstack(feedback).T
                    fp.write("trees: " + str(n) + " depth:" + str(d) + " adaptive:" + str(a) + '\n')
                    fp.write(str(res))
                    fp.write('\n')
                    fp.write(str(feed))
                    fp.write('\n')
        fp.close()
    # for af in filelist:
    #     datecontent = pd.read_csv(datadir + af).values
    #     HST(datecontent)

# !pip install japanize_matplotlib

import matplotlib.pyplot as plt
import numpy as np
import math
# import japanize_matplotlib
import torch
from params import params_set
import os

from params import params_set

def valid_loss():
    # normalの結果
    [normal_x, normal_y] = get_valid_loss_array("./models/normal/param_set-0")

    for param_name in params_set:
        if param_name == "normal":
            continue
        print(f'-------- [{param_name}] --------')
        out_dir = "./img/valid_loss"
        file_name = param_name

        [x1, y1] = get_valid_loss_array("./models/" + param_name + "/param_set-0")
        [x2, y2] = get_valid_loss_array("./models/" + param_name + "/param_set-1")
        [x3, y3] = get_valid_loss_array("./models/" + param_name + "/param_set-2")

        if param_name == "patch_size":
            label2 = param_name + " = " + str(params_set["normal"][0][param_name])
            label0 = param_name + " = " + str(params_set[param_name][0][param_name])
            label1 = param_name + " = " + str(params_set[param_name][1][param_name])
            label3 = param_name + " = " + str(params_set[param_name][2][param_name])
            save_graph(x1, y1, x2, y2, normal_x, normal_y, x3, y3, label0, label1, label2, label3, out_dir, file_name, 85, 200)
        else:
            label0 = param_name + " = " + str(params_set["normal"][0][param_name])
            label1 = param_name + " = " + str(params_set[param_name][0][param_name])
            label2 = param_name + " = " + str(params_set[param_name][1][param_name])
            label3 = param_name + " = " + str(params_set[param_name][2][param_name])
            save_graph(normal_x, normal_y, x1, y1, x2, y2, x3, y3, label0, label1, label2, label3, out_dir, file_name, 85, 200)

def acculacy_img():
    # normalの結果
    [normal_x, normal_y] = get_acculacy_array("./results/normal/param_set-0")

    for param_name in params_set:
        if param_name == "normal":
            continue
        print(f'-------- [{param_name}] --------')
        out_dir = "./img/acculacy"
        file_name = param_name

        [x1, y1] = get_acculacy_array("./results/" + param_name + "/param_set-0")
        [x2, y2] = get_acculacy_array("./results/" + param_name + "/param_set-1")
        [x3, y3] = get_acculacy_array("./results/" + param_name + "/param_set-2")

        if param_name == "patch_size":
            label2 = param_name + " = " + str(params_set["normal"][0][param_name])
            label0 = param_name + " = " + str(params_set[param_name][0][param_name])
            label1 = param_name + " = " + str(params_set[param_name][1][param_name])
            label3 = param_name + " = " + str(params_set[param_name][2][param_name])
            save_graph(x1, y1, x2, y2, normal_x, normal_y, x3, y3, label0, label1, label2, label3, out_dir, file_name, 0.15, 0.7, True)
        else:
            label0 = param_name + " = " + str(params_set["normal"][0][param_name])
            label1 = param_name + " = " + str(params_set[param_name][0][param_name])
            label2 = param_name + " = " + str(params_set[param_name][1][param_name])
            label3 = param_name + " = " + str(params_set[param_name][2][param_name])
            save_graph(normal_x, normal_y, x1, y1, x2, y2, x3, y3, label0, label1, label2, label3, out_dir, file_name, 0.15, 0.7, True)

    
def get_valid_loss_array(model_dir):
    """
        ディレクトリを指定して，内部のモデルのindexとvalid_lossの配列を返す
    """
    x = []
    y = []
    for i in range(len(os.listdir(model_dir))):
        file_path = model_dir + "/" + str(i) + ".cpt"
        cpt = torch.load(file_path)
        valid_loss = cpt['valid_loss'].item()
        x.append(i)
        y.append(valid_loss)
    
    return [x, y]

def get_acculacy_array(model_dir):
    """
        ディレクトリを指定して，内部のモデルのindexとaccuracyの配列を返す
    """
    x = []
    y = []
    for i in range(30):
        file_name = str(i) + ".cpt"
        if file_name in os.listdir(model_dir):
            file_path = model_dir + "/" + str(i) + ".cpt"
            cpt = torch.load(file_path)
            accuracy = cpt['accuracy'].item()
            x.append(i)
            y.append(accuracy)
    
    return [x, y]

def save_graph(x0, y0, x1, y1, x2, y2, x3, y3, label0, label1, label2, label3, out_dir, file_name, y_lim_min, y_lim_max, need_marker=False):
    #オフセット表現をやめる
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

    if need_marker:
        plt.plot(x0, y0, marker="s",label=label0)
        plt.plot(x1, y1, marker="s",label=label1)
        plt.plot(x2, y2, marker="s",label=label2)
        plt.plot(x3, y3, marker="s",label=label3)
    else:
        plt.plot(x0, y0, label=label0)
        plt.plot(x1, y1, label=label1)
        plt.plot(x2, y2, label=label2)
        plt.plot(x3, y3, label=label3)

    plt.xlabel("epoch")
    plt.ylabel("valid loss")

    #軸の範囲を指定
    plt.ylim(y_lim_min, y_lim_max)

    #凡例をつける場合
    plt.legend()

    #軸ラベルがみ切れる場合の対処法
    plt.tight_layout()

    # 出力先のディレクトリを作成
    p = out_dir
    if not os.path.isdir(p):
        os.makedirs(p)

    plt.savefig(os.path.join(p, file_name + ".png"), format="png", dpi=200)

    plt.clf()

valid_loss()
acculacy_img()
# -*- coding: utf-8 -*-
# EditTime  : 2019/12/4 20:24
# Author    : Of yue
# File      : trans_img2np.py
# Intro     : 将图片转化为numpy矩阵

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_PATH = "hasyv2/"
IMAGE_LABELS = "hasyv2/hasy-data-labels.csv"

OUTPUT_PATH_X = "output//data_x"
OUTPUT_PATH_Y = "output//data_y"

# 要转化为矩阵的数据
trans_list = [r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\zeta", r"\eta", r"\theta", r"\iota", r"\kappa",
              r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\rho", r"\sigma", r"\tau", r"\upsilon", r"\phi", r"\chi",
              r"\psi", r"\omega", r"\vartheta", r"\varrho"]

data_all = pd.read_csv(IMAGE_LABELS)
data_all["latex"] = data_all["latex"].str.strip()

# 从所有数据中选择希腊字母
data_to_trans = data_all[data_all.latex.isin(trans_list)]
print(data_to_trans.shape)

# 将图片转化为像素矩阵
data_x = []
data_y = []

for i, image in data_all.iterrows():
    im = Image.open(IMAGE_PATH + image.path).convert("L")
    im = np.asarray(im)

    data_x.append(im)
    data_y.append(image.symbol_id)

data_x = np.array(data_x)
data_y = np.array(data_y)

# 存储所得numpy数组
np.save(OUTPUT_PATH_X, data_x)
np.save(OUTPUT_PATH_Y, data_y)

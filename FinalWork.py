# -*- coding: utf-8 -*-
# EditTime  : 2019/12/5 13:52
# Author    : Of yue
# File      : FinalWork.py
# Intro     : 大作业脚本

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


class FinalWork:
    def __init__(self, data_x_path, data_y_path, label_path):
        self.data_x = np.load(data_x_path)
        self.data_y = np.load(data_y_path)
        self.symbols = pd.read_csv(label_path)[["symbol_id", "latex"]]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_x, self.data_y, test_size=0.2)

        # 预处理训练集与测试集
        pixels = self.x_train.shape[1] * self.x_train.shape[2]
        self.x_train = self.x_train.reshape(self.x_train.shape[0], pixels)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], pixels)
        self.x_train, self.x_test = self.x_train / 255., self.x_test / 255.

        # 测试集标签种数
        self.unique_test, self.counts_test = np.unique(self.y_test, return_counts=True)

        # 初始化pca
        self.pca_list = []

    def label2symbol(self, label):
        """
        转化标签和符号
        :param label: 标签值
        :return: 对应符号的latex代码
        """
        index = self.symbols.loc[self.symbols["symbol_id"] == label]
        if not index.empty:
            return "$" + str(index["latex"].values[0]) + "$"
        else:
            return "None"

    def show_data_image(self, pix_data, label=None):
        """
        将像素矩阵转化为图片显示出来
        :param pix_data: 像素矩阵
        :param label: 图片标签，可以没有
        :return: 无
        """
        fig, p = plt.subplots()
        p.imshow(pix_data, cmap="gray")
        p.title.set_text("value:" + self.label2symbol(label))

    def evaluate_model(self, model, use_pca=False, return_confusion_matrix=False):
        """
        评估模型准确度，计算混淆矩阵
        :param model: 模型
        :param use_pca: 选择是否使用PCA降维
        :param return_confusion_matrix: 选择是否计算混淆矩阵
        :return: 模型，分数，最好参数的PCA模型(没有则为0)，混淆矩阵(没有则为0)
        """
        pca, cm = 0, 0
        if use_pca:
            self.generate_pca_list()
            model, score, pca, index = self.find_best_pca(model)
            if return_confusion_matrix:
                try:
                    cm = confusion_matrix(self.y_test, model.predict(self.pca_list[index][1]), labels=self.unique_test)
                except Exception:
                    print("predict shape: {}".format(self.pca_list[index][1].shape))
                    print("trained shape: {}".format(self.pca_list[index][0].shape))
                    print("index: {}".format(index))
                    print(model)
        else:
            model.fit(self.x_train, self.y_train)
            score = model.score(self.x_test, self.y_test)
            if return_confusion_matrix:
                cm = confusion_matrix(self.y_test, model.predict(self.x_test), labels=self.unique_test)
        return model, score, pca, cm

    def generate_pca(self, n):
        """
        生成参数为n的PCA模型，对训练集和测试集预处理
        :param n:
        :return:
        """
        pca = PCA(n_components=n)
        pca_x_train = pca.fit_transform(self.x_train)
        pca_x_test = pca.transform(self.x_test)
        return pca_x_train, pca_x_test, pca

    def generate_pca_list(self):
        if not self.pca_list:
            self.pca_list = [self.generate_pca(n) for n in np.arange(0.1, 1, 0.1)]

    def find_best_pca(self, model):
        """
        取不同的PCA模型的n值，选出最好的(模型准确率最高的)
        :param model:
        :return:
        """
        pca_x_train, pca_x_test, pca = self.pca_list[0]
        model.fit(pca_x_train, self.y_train)
        score = model.score(pca_x_test, self.y_test)
        model_best, score_best, pca_best, index_best = model, score, pca, 0
        for index, (pca_x_train, pca_x_test, pca) in enumerate(self.pca_list[1:]):
            model.fit(pca_x_train, self.y_train)
            score = model.score(pca_x_test, self.y_test)
            if score > score_best:
                index_best = index

        pca_x_train, pca_x_test, pca = self.pca_list[index_best]
        model.fit(pca_x_train, self.y_train)
        score = model.score(pca_x_test, self.y_test)
        model_best, score_best, pca_best = model, score, pca
        return model_best, score_best, pca_best, index_best

    def draw_confusion_matrix(self, cm, labels=None):
        """
        混淆矩阵可视化
        :param cm: 混淆矩阵
        :param labels: 标签列表
        :return:
        """
        if not labels:
            labels = self.unique_test
        # 绘制混淆矩阵
        image_size = labels.shape[0]
        unique_symbols = [self.label2symbol(label) for label in labels]
        plt.figure(figsize=(image_size, image_size))

        ax = plt.gca()
        ax.matshow(cm, cmap=plt.cm.gray)

        locator = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(locator)
        ax.yaxis.set_major_locator(locator)

        ax.set_xticks(np.arange(0.3, 0.3 + image_size, 1))
        ax.set_yticks(np.arange(0.2, 0.2 + image_size, 1))
        ax.set_xticklabels(unique_symbols)
        ax.set_yticklabels(unique_symbols)
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(labelsize=30)
        plt.show()

    def draw_error_rate_matrix(self, cm):
        """
        错误率矩阵可视化
        :param cm: 混淆矩阵
        :return:
        """
        symbol_nums = cm.sum(axis=1, keepdims=True)
        error_rate = cm / symbol_nums
        # 对角线置0
        np.fill_diagonal(error_rate, 0)
        self.draw_confusion_matrix(error_rate)

    def predict_png(self, mod, png_path, pca=None):
        """
        利用模型对图片进行预测，输出预测后的值
        :param mod:
        :param png_path:
        :param pca: 如果模型使用了PCA，则输入相对应的PCA
        :return:
        """
        from PIL import Image
        im = Image.open(png_path).convert("L")
        im = np.asarray(im)
        self.show_data_image(im)
        im = im.reshape(1, 1024) / 255.

        if pca:
            im = pca.transform(im)
        pre = mod.predict(im)
        return pre[0]

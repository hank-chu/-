# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:44:52 2021

@author: USER
"""

# 匯入所需的庫
from fastai import *
from fastai.vision import *
from fastai.vision.all import *

#使用 cnn_learner 創建一個 CNN 模型,使用 ResNet101 作為基礎架構,並設置準確度作為評估指標。
#to_fp16() 用於將模型轉換為半精度運算,這可以加快訓練速度,特別是在有GPU的情況下。
learn = cnn_learner(data, models.resnet101, metrics=accuracy).to_fp16()

#找到最佳學習率,這是訓練前的一個重要步驟,幫助確定最適合的學習率。
learn.lr_find()

#使用 one cycle 方法進行訓練,訓練 2 個週期,並將初始學習率設置為 0.01。
learn.fit_one_cycle(2, 1e-2)

#將預計算設置為 False,這意味著將解凍所有層以進行進一步的微調。
learn.precompute = False

#重新找到最佳學習率,這是因為模型現在已經解凍。
learn.lr_find()

#進行進一步的訓練,這次訓練 4 個週期,使用一個學習率範圍。
#lr_max=slice(1e-8, 1e-4) 表示學習率將從 1e-8 緩慢增加到 1e-4。
learn.fit_one_cycle(4, lr_max=slice(1e-8, 1e-4))

#重複上述步驟直到模型達到最佳效果,這是一個常見的訓練策略。
#您可以根據驗證集的損失和準確度來判斷模型的飽和狀態。

#設置預計算為 True,這是為了再次凍結模型的層,以便進行微調。
learn.precompute = True

#凍結模型的層以避免過擬合,這樣可以提高訓練效率。
learn.freeze()

#儲存訓練好的模型以備將來使用。
#modelpath = "path_to_save_model"
#learn.export(modelpath)

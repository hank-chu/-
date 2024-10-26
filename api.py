# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 03:19:23 2021

@author: USER
"""

from argparse import ArgumentParser
import base64
from datetime import datetime
import hashlib
import time

import cv2
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

from fastai import *
from fastai.vision import *
from fastai.vision.all import *
warnings.filterwarnings(action='ignore')

# 設定資料路徑,讀取圖像資料集
datapath = r'/home/USER/new_data_does_not_have_isnull'
data = ImageDataLoaders.from_folder(datapath, valid_pct=0.2, size=100*100)
learn = cnn_learner(data, models.resnet101, metrics=accuracy).to_fp16()  
learn.load(r'testmodel5')  # 載入預訓練模型

app = Flask(__name__)  # 建立 Flask 應用
app.config['JSON_AS_ASCII'] = False  # 設定JSON編碼,不使用ASCII

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'reinlikes@gmail.com'  # 設定隊長的電子郵件
SALT = 'aipineapple'  # 設定用於生成UUID的鹽
#########################################

def generate_server_uuid(input_string):
    """ 
    生成唯一的server_uuid

    @param:
        input_string (str): 要編碼的資訊
    @returns:
        server_uuid (str): 生成的唯一server_uuid
    """
    s = hashlib.sha256()  # 創建SHA256哈希對象
    data = (input_string + SALT).encode("utf-8")  # 將輸入字符串與鹽結合
    s.update(data)  # 更新哈希對象
    server_uuid = s.hexdigest()  # 生成十六進制格式的UUID
    return server_uuid

def base64_to_binary_for_cv2(image_64_encoded):
    """ 
    將base64編碼的圖像轉換為cv2可用的numpy.ndarray格式

    @param:
        image_64_encode(str): base64編碼的圖像字符串
    @returns:
        image(numpy.ndarray): 轉換後的圖像
    """
    img_base64_binary = image_64_encoded.encode("utf-8")  # 將base64字符串編碼
    img_binary = base64.b64decode(img_base64_binary)  # 解碼base64字符串
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)  # 將二進制數據解碼為圖像
    picture_path = r"/home/aipineapple0606/picture"  # 設定圖片儲存路徑
    name = time.time()  # 使用當前時間作為文件名
    filename = picture_path + "\\" + str(name) + ".jpg"  # 設定圖片文件名
    cv2.imwrite(filename, image)  # 儲存圖片
    return image  # 返回圖像

def process_image(img, min_side):
    """ 
    處理圖像,調整大小並填充

    @param:
        img (numpy.ndarray): 要處理的圖像
        min_side (int): 最小邊長
    @returns:
        pad_img (numpy.ndarray): 填充後的圖像
    """
    size = img.shape  # 獲取圖像的尺寸
    h, w = size[0], size[1]
    # 將長邊縮放至min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)  # 計算新尺寸
    resize_img = cv2.resize(img, (new_w, new_h))  # 調整圖像大小
    # 根據新尺寸填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    else:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2

    # 使用邊界擴展填充圖像
    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[255, 255, 255])  
    return pad_img  # 返回填充後的圖像

def Prob_judgment(img):
    """ 
    判斷預測結果的概率

    @param:
        img (numpy.ndarray): 要預測的圖像
    @returns:
        word (str): 預測的單詞或 'isnull'
    """
    ten10 = learn.predict(img)  # 使用模型進行預測
    word = ten10[0]  # 獲取預測的單詞
    print(ten10[2][ten10[1]])  # 打印預測概率
    if ten10[2][ten10[1]] > 0.5:
        return word  # 如果概率大於0.5,返回預測的單詞
    else:
        return 'isnull'  # 否則返回 'isnull'

def predict(image):
    """ 
    預測圖像的結果

    @param:
        image (numpy.ndarray): 要預測的圖像
    @returns:
        prediction (str): 預測的單詞
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 將圖像轉換為灰階

    kernel = np.ones((2, 2), np.uint8)  # 創建膨脹核
    ret, img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 應用自適應閾值

    sp = img.shape  # 獲取圖像的形狀
    sz1 = sp[0]  # 獲取圖像高度
    sz2 = sp[1]  # 獲取圖像寬度

    # 橫線刪除    
    for i in range(0, sz1):
        if sum(img[i]) < len(img[i]) * 20:
            for j in range(len(img[i])):
                img[i][j] = 255  # 將行像素設為白色

    # 直線刪除
    for j in range(len(img[i])):
        c = 0
        for i in range(0, sz1):       
            c += img[i][j]
        if c < sz1 * 30:
            for i in range(0, sz1):       
                img[i][j] = 255  # 將列像素設為白色
    
    # 預處理圖像
    img = process_image(img, 100)  # 處理圖像,調整大小
    word = Prob_judgment(img)  # 進行預測
    return word  # 返回預測結果

@app.route('/predict', methods=['POST'])
def predict_api():
    """ 
    API接口,用於處理預測請求

    @return:
        result (json): 包含預測結果的JSON響應
    """
    data = request.json  # 獲取請求的JSON數據
    uuid = generate_server_uuid(CAPTAIN_EMAIL)  # 生成唯一的server_uuid
    image_64_encoded = data['data']  # 獲取base64編碼的圖像數據
    image = base64_to_binary_for_cv2(image_64_encoded)  # 轉換圖像格式
    word = predict(image)  # 預測圖像
    result = {'uuid': uuid, 'data': word}  # 構建響應結果
    return jsonify(result)  # 返回JSON響應

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 啟動Flask應用

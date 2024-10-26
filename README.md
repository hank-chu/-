# **中文手寫圖片辨識**
機器學習：辨識中文手寫圖片辨識  
Machine Learning: Recognizing Chinese Handwritten Characters

## **介紹**
此專案記錄參加玉山人工智慧挑戰賽2021夏季賽的過程，目的是從影像中辨識中文手寫字。專案透過圖片清理、數據增強並自行生成訓練數據，利用 fastai 建立了一個 CNN 模型。最終，我們將此模型部署於 Google Cloud Platform 上，提供即時辨識服務。

## **預期模型應用**
中文手寫影像辨識可用於各種場景，如教育評估系統、筆跡分析、文檔數字化等。此模型可幫助使用者在不同情境下實現中文手寫的自動化識別，從而提高文檔處理效率。對於有大量手寫數據需求的公司及研究人員，該模型將成為一項實用的工具。

## **技術挑戰與解決方案**
- 數據清理：實施圖片灰階、二值化等前處理，確保輸入一致性。
- 數據增強：使用隨機字體和旋轉角度生成多樣化數據，提高模型的泛化能力。
- 模型訓練：通過 CNN 架構設計與調參，最終獲得約 90% 的準確率。

## **相關技術**
- `Python`、`fastai`、`OpenCV`、`Google Cloud Platform`、`Flask API`

## **使用方式**
#### 圖片前處理：adjust_picture.py
此腳本用於調整圖片大小、灰階轉換和二值化處理，確保圖像清晰且適合模型輸入。

#### 生成訓練數據：generate_training_data.py
生成多樣化的訓練數據，增加模型訓練的泛化性。可自行調整目標文字，再通過不同字體和隨機增強生成圖像。

#### 建立 CNN 模型：build_model.py
使用 fastai 建立 CNN 模型。此步驟包括模型的訓練、調參及最終的模型儲存，以便後續進行推論。

#### API 即時辨識：api.py
使用 Flask 框架構建的 API，實現圖像辨識的即時服務。此 API 可用於接收圖片並返回辨識結果，方便用於多種應用場景。

# Big Data Analysis

這個專案用於分析溫室環境數據，包括：
- 溫度
- 濕度
- 二氧化碳濃度
- 土壤濕度
- 光照狀態

## 功能
- 讀取 Excel 格式的數據
- 進行基本的描述性統計分析
- 生成數據視覺化圖表：
  - 溫度與濕度關係散點圖
  - 光照條件與二氧化碳濃度關係箱型圖
  - 二氧化碳濃度分布直方圖
- 機器學習分析：
  - 使用溫度和濕度預測土壤濕度的線性回歸模型
  - 模型性能評估（R²分數、RMSE）
  - 3D可視化預測結果

## 需求套件
- pandas
- matplotlib
- seaborn
- openpyxl
- scikit-learn
- numpy

## 環境設置
1. 確保使用正確的 Python 環境：
   ```bash
   # 使用虛擬環境中的 Python
   c:\users\p2451fav\.virtualenvs\fastapi-horuss-one-fgyxkgx9\Scripts\python.exe
   ```

2. 安裝所需套件：
   ```bash
   python -m pip install pandas matplotlib seaborn scikit-learn openpyxl
   ```

3. VSCode 設置：
   - 按 F1 或 Ctrl+Shift+P
   - 輸入 "Python: Select Interpreter"
   - 選擇包含 "fastapi-horuss-one-fgyxkgx9" 的虛擬環境

## 執行說明
運行程式將會：
1. 讀取並顯示數據基本統計資訊
2. 生成溫度與濕度關係散點圖
3. 顯示光照條件與二氧化碳濃度的關係圖
4. 建立土壤濕度預測模型並評估效能
5. 生成 3D 預測結果視覺化圖表

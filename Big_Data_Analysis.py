"""
綠豆種植環境數據分析程式
本程式分析綠豆種植環境的各項參數數據，包括溫度、濕度、光照、二氧化碳濃度等，
並建立預測土壤濕度的線性回歸模型。

主要功能：
1. 數據讀取和基本統計分析
2. 環境參數之間關係的視覺化
3. 建立土壤濕度預測模型
4. 模型評估和視覺化
"""

# 導入必要的函式庫
import pandas as pd  # 用於數據處理和分析
import matplotlib.pyplot as plt  # 用於數據視覺化
import seaborn as sns  # 用於統計數據視覺化
from matplotlib.font_manager import FontProperties  # 用於設定中文字型
from sklearn.model_selection import train_test_split  # 用於分割訓練集和測試集
from sklearn.linear_model import LinearRegression  # 線性回歸模型
from sklearn.metrics import r2_score, mean_squared_error  # 模型評估指標
import numpy as np  # 用於數值計算

# 設定中文字型以正確顯示中文標籤
font = FontProperties(fname=r'C:\Windows\Fonts\msjh.ttc', size=12)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 讀取數據集
df = pd.read_excel('green_bean_data_updated.xlsx')  # 讀取綠豆種植環境數據

# 顯示數據集的前幾行，用於初步檢視數據
print(df.head())

# 計算並顯示基本描述性統計量（如均值、標準差、最大最小值等）
print(df.describe())

# 繪製溫度與濕度的散點圖，分析兩者關係
plt.figure(figsize=(10, 6))
sns.scatterplot(x='溫度 (°C)', y='濕度 (%)', data=df)
plt.title('溫度與濕度的關係', fontproperties=font)
plt.xlabel('溫度 (°C)', fontproperties=font)
plt.ylabel('濕度 (%)', fontproperties=font)
plt.show()

# 使用箱形圖分析光照條件對二氧化碳濃度的影響
plt.figure(figsize=(10, 6))
sns.boxplot(x='光照 (Yes/No)', y='二氧化碳濃度 (ppm)', data=df)
plt.title('光照條件下的二氧化碳濃度分布', fontproperties=font)
plt.xlabel('是否有光照', fontproperties=font)
plt.ylabel('二氧化碳濃度 (ppm)', fontproperties=font)
plt.show()

# 建立土壤濕度的多變量線性回歸模型
from sklearn.preprocessing import StandardScaler

# 準備特徵(X)和目標變數(y)
X = df[['溫度 (°C)', '濕度 (%)']].copy()  # 選擇溫度和濕度作為特徵
y = df['土壤濕度 (%)'].copy()  # 目標變數為土壤濕度

# 使用StandardScaler進行特徵標準化，使不同尺度的特徵具有可比性
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=['溫度 (標準化)', '濕度 (標準化)'])

# 將數據分為訓練集(80%)和測試集(20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 建立並訓練線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 使用測試集進行預測
y_pred = model.predict(X_test)

# 計算模型評估指標
r2 = r2_score(y_test, y_pred)  # R平方值，衡量模型解釋變異的程度
mse = mean_squared_error(y_test, y_pred)  # 均方誤差
rmse = np.sqrt(mse)  # 均方根誤差

# 輸出模型評估結果
print("\n回歸模型結果：")
print(f"R² 分數：{r2:.4f}")
print(f"均方根誤差 (RMSE)：{rmse:.4f}")
print(f"標準化溫度係數：{model.coef_[0]:.4f}")
print(f"標準化濕度係數：{model.coef_[1]:.4f}")
print(f"截距：{model.intercept_:.4f}")

# 計算並顯示特徵重要性
importance = abs(model.coef_)
for name, imp in zip(['溫度', '濕度'], importance):
    print(f"{name}重要性：{imp:.4f}")

# 繪製實際值vs預測值的散點圖，用於視覺化模型預測效果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('土壤濕度預測模型的實際值vs預測值', fontproperties=font)
plt.xlabel('實際土壤濕度 (%)', fontproperties=font)
plt.ylabel('預測土壤濕度 (%)', fontproperties=font)
plt.show()

# 建立3D預測曲面圖，展示溫度、濕度與土壤濕度的關係
# 生成網格數據點
temp_min, temp_max = df['溫度 (°C)'].min(), df['溫度 (°C)'].max()
humidity_min, humidity_max = df['濕度 (%)'].min(), df['濕度 (%)'].max()

# 在溫度和濕度範圍內創建均勻分布的點
temp_range = np.linspace(temp_min, temp_max, 50)
humidity_range = np.linspace(humidity_min, humidity_max, 50)
temp_mesh, humidity_mesh = np.meshgrid(temp_range, humidity_range)

# 準備網格數據並進行標準化
X_mesh = np.column_stack((temp_mesh.ravel(), humidity_mesh.ravel()))
X_mesh_scaled = scaler.transform(X_mesh)
soil_moisture_pred = model.predict(X_mesh_scaled)

# 繪製3D預測曲面
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(temp_mesh, humidity_mesh, 
                      soil_moisture_pred.reshape(temp_mesh.shape),
                      cmap='viridis', alpha=0.8)
ax.set_title('溫度、濕度與預測土壤濕度的關係', fontproperties=font, y=1.1)
ax.set_xlabel('溫度 (°C)', fontproperties=font)
ax.set_ylabel('濕度 (%)', fontproperties=font)
ax.set_zlabel('預測土壤濕度 (%)', fontproperties=font)
plt.colorbar(surf, ax=ax, label='土壤濕度 (%)')
plt.show()

# 分析二氧化碳濃度分布
plt.figure(figsize=(10, 6))
sns.histplot(df['二氧化碳濃度 (ppm)'], kde=True, color='blue')
plt.title('二氧化碳濃度分布', fontproperties=font)
plt.xlabel('二氧化碳濃度 (ppm)', fontproperties=font)
plt.ylabel('頻數', fontproperties=font)
plt.show()

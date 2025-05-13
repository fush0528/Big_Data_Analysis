import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 設定中文字型
font = FontProperties(fname=r'C:\Windows\Fonts\msjh.ttc', size=12)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 讀取 Excel 檔案
df = pd.read_excel('green_bean_data_updated.xlsx')  # 或者使用完整路徑

# 顯示前幾行數據
print(df.head())

# 基本描述性統計分析
print(df.describe())

# 繪製溫度與濕度的關係圖
plt.figure(figsize=(10, 6))
sns.scatterplot(x='溫度 (°C)', y='濕度 (%)', data=df)
plt.title('溫度與濕度的關係', fontproperties=font)
plt.xlabel('溫度 (°C)', fontproperties=font)
plt.ylabel('濕度 (%)', fontproperties=font)
plt.show()

# 分析光照時間與二氧化碳濃度的關係
plt.figure(figsize=(10, 6))
sns.boxplot(x='光照 (Yes/No)', y='二氧化碳濃度 (ppm)', data=df)
plt.title('光照條件下的二氧化碳濃度分布', fontproperties=font)
plt.xlabel('是否有光照', fontproperties=font)
plt.ylabel('二氧化碳濃度 (ppm)', fontproperties=font)
plt.show()

# 建立土壤濕度的回歸模型
# 準備數據
X = df[['溫度 (°C)', '濕度 (%)']]
y = df['土壤濕度 (%)']

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立和訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

# 進行預測
y_pred = model.predict(X_test)

# 評估模型
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 打印模型結果
print("\n回歸模型結果：")
print(f"R² 分數：{r2:.4f}")
print(f"均方根誤差 (RMSE)：{rmse:.4f}")
print(f"溫度係數：{model.coef_[0]:.4f}")
print(f"濕度係數：{model.coef_[1]:.4f}")
print(f"截距：{model.intercept_:.4f}")

# 視覺化預測結果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('土壤濕度預測模型的實際值vs預測值', fontproperties=font)
plt.xlabel('實際土壤濕度 (%)', fontproperties=font)
plt.ylabel('預測土壤濕度 (%)', fontproperties=font)
plt.show()

# 繪製預測曲面圖
temp_range = np.linspace(X['溫度 (°C)'].min(), X['溫度 (°C)'].max(), 50)
humidity_range = np.linspace(X['濕度 (%)'].min(), X['濕度 (%)'].max(), 50)
temp_mesh, humidity_mesh = np.meshgrid(temp_range, humidity_range)
X_mesh = np.column_stack((temp_mesh.ravel(), humidity_mesh.ravel()))
soil_moisture_pred = model.predict(X_mesh)

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

# 繪製二氧化碳濃度的直方圖
plt.figure(figsize=(10, 6))
sns.histplot(df['二氧化碳濃度 (ppm)'], kde=True, color='blue')
plt.title('二氧化碳濃度分布', fontproperties=font)
plt.xlabel('二氧化碳濃度 (ppm)', fontproperties=font)
plt.ylabel('頻數', fontproperties=font)
plt.show()

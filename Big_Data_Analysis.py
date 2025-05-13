import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

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

# 繪製二氧化碳濃度的直方圖
plt.figure(figsize=(10, 6))
sns.histplot(df['二氧化碳濃度 (ppm)'], kde=True, color='blue')
plt.title('二氧化碳濃度分布', fontproperties=font)
plt.xlabel('二氧化碳濃度 (ppm)', fontproperties=font)
plt.ylabel('頻數', fontproperties=font)
plt.show()

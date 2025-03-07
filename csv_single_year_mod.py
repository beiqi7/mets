import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from matplotlib.font_manager import FontProperties

# 设置中文字体和绘图风格
font = FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# 读取数据
df = pd.read_csv('1990-2023_wj.csv')

# 随机抽取一条数据进行测试
random_row = df.iloc[random.randint(0, len(df)-1)]
print("\n随机测试数据:")
print(f"日期: {random_row['日期']}")
print(f"最低气温: {random_row['最低气温']}°C")
print("-" * 50)

# 将日期列转换为datetime格式
df['日期'] = pd.to_datetime(df['日期'])

# 分割数据为基准期(1990-2020)和检测期(2021-2023)
base_period = df[(df['日期'].dt.year >= 1990) & (df['日期'].dt.year <= 2020)]
test_period = df[(df['日期'].dt.year >= 2021) & (df['日期'].dt.year <= 2023)]

# 计算每个日历日的阈值
def calculate_threshold(date, base_data):
    # 获取月和日
    month = date.month
    day = date.day
    
    # 创建前后7天的日期范围
    dates = []
    for year in range(1990, 2021):
        # 创建当年的目标日期
        target_date = datetime(year, month, day)
        # 添加前后7天的日期
        for delta in range(-7, 8):
            window_date = target_date + timedelta(days=delta)
            dates.append(window_date)
    
    # 获取这些日期对应的温度值
    temp_values = base_data[base_data['日期'].isin(dates)]['最低气温']
    
    # 计算90百分位阈值
    return np.percentile(temp_values, 90)

# 为检测期的每一天计算阈值
test_period = test_period.copy()
test_period['阈值'] = test_period['日期'].apply(lambda x: calculate_threshold(x, base_period))

# 找出超过阈值的日期
test_period['超阈值'] = test_period['最低气温'] >= test_period['阈值']
test_period['超温程度'] = np.where(test_period['超阈值'], 
                               test_period['最低气温'] - test_period['阈值'], 
                               0)

# 识别连续的热浪事件
def find_continuous_heatwaves(data):
    heatwave_events = []
    current_event = []
    
    # 按日期排序
    data = data.sort_values('日期')
    
    for idx, row in data.iterrows():
        if len(current_event) == 0:
            current_event.append(row)
        else:
            # 检查日期是否连续
            prev_date = current_event[-1]['日期']
            curr_date = row['日期']
            days_diff = (curr_date - prev_date).days
            
            if days_diff == 1:  # 日期连续
                current_event.append(row)
            else:  # 日期不连续
                if len(current_event) >= 3:  # 如果当前事件持续3天及以上
                    heatwave_events.append(pd.DataFrame(current_event))
                current_event = [row]  # 开始新的事件
    
    # 处理最后一个事件
    if len(current_event) >= 3:
        heatwave_events.append(pd.DataFrame(current_event))
    
    return heatwave_events

# 找出连续的热浪事件
heatwave_events = find_continuous_heatwaves(test_period[test_period['超阈值']])
heatwave_dates = pd.concat(heatwave_events) if heatwave_events else pd.DataFrame()

# 创建输出文件
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'夜间热浪分析结果_{current_time}.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("基准期(1990-2020)阈值说明:\n")
    f.write("采用15天滑动窗口计算每个日历日的90百分位阈值\n")
    f.write("每个日期的阈值基于该日期前后7天在1990-2020年间的所有温度值计算\n\n")
    f.write("夜间热浪事件(连续3天及以上超过阈值):\n")
    
    # 按年份分组输出热浪事件
    for year in sorted(heatwave_dates['日期'].dt.year.unique()):
        year_events = [event for event in heatwave_events if event['日期'].dt.year.iloc[0] == year]
        f.write(f"\n{year}年热浪事件:\n")
        for event in year_events:
            f.write("\n持续时间: {} 天\n".format(len(event)))
            f.write(event[['日期', '最低气温', '阈值', '超温程度']].to_string(index=False))
            f.write("\n")
    
    # 年度统计
    yearly_events = heatwave_dates.groupby(heatwave_dates['日期'].dt.year).agg({
        '日期': 'count',
        '超温程度': ['mean', 'max']
    })
    yearly_events.columns = ['热浪天数', '平均超温程度', '最大超温程度']
    f.write("\n年度统计(基于1990-2020基准期):\n")
    f.write(yearly_events.to_string())

# 创建更直观的图表
plt.figure(figsize=(15, 12))

# 子图1：热浪发生频次和持续时间
ax1 = plt.subplot(3, 1, 1)
sns.barplot(x=yearly_events.index, y=yearly_events['热浪天数'], 
            color='orangered', alpha=0.7)
plt.title('2021-2023年夜间热浪频次统计(基准期:1990-2020)', fontproperties=font, fontsize=12, pad=15)
plt.xlabel('年份', fontproperties=font)
plt.ylabel('热浪天数', fontproperties=font)
for i, v in enumerate(yearly_events['热浪天数']):
    ax1.text(i, v, str(v), ha='center', va='bottom', fontproperties=font)

# 子图2：热浪强度时间序列
ax2 = plt.subplot(3, 1, 2)
for event in heatwave_events:
    plt.plot(event['日期'], event['超温程度'], 
             marker='o', linestyle='-', color='red', alpha=0.6)
plt.title('夜间热浪强度时间序列(基准期:1990-2020)', fontproperties=font, fontsize=12, pad=15)
plt.xlabel('日期', fontproperties=font)
plt.ylabel('超过阈值温度（°C）', fontproperties=font)
plt.grid(True, alpha=0.3)

# 子图3：热浪强度月度分布
ax3 = plt.subplot(3, 1, 3)
heatwave_dates['月份'] = heatwave_dates['日期'].dt.month
monthly_intensity = heatwave_dates.groupby('月份')['超温程度'].agg(['mean', 'count'])
sns.boxplot(data=heatwave_dates, x='月份', y='超温程度', color='orange')
plt.title('夜间热浪强度月度分布(基准期:1990-2020)', fontproperties=font, fontsize=12, pad=15)
plt.xlabel('月份', fontproperties=font)
plt.ylabel('超过阈值温度（°C）', fontproperties=font)

plt.tight_layout()
plt.savefig(f'夜间热浪分析图表_{current_time}.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"分析结果已保存至: {output_file}")
print(f"图表已保存至: 夜间热浪分析图表_{current_time}.png")

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, KFold
import pykrige
import random
from datetime import datetime
import matplotlib.font_manager as fm
import platform
import os

def load_data(nc_file_path, observed_data_path):
    """
    加载克里金插值结果和观测数据
    
    参数:
    nc_file_path: 克里金插值生成的nc文件路径
    observed_data_path: 原始观测数据的路径
    
    返回:
    kriged_data: 插值后的数据
    observed_data: 观测数据
    """
    # 加载nc文件
    kriged_data = xr.open_dataset(nc_file_path)
    
    # 检查时间索引是否单调
    if 'time' in kriged_data.dims:
        time_values = kriged_data.time.values
        try:
            # 将时间差转换为数值类型进行比较
            time_diffs = np.diff(time_values).astype('timedelta64[ns]').astype(np.int64)
            if not (np.all(time_diffs > 0) or np.all(time_diffs < 0)):
                print("警告：时间索引不是单调的，尝试排序...")
                # 对时间索引进行排序
                kriged_data = kriged_data.sortby('time')
                print("时间索引已排序")
        except Exception as e:
            print(f"警告：检查时间索引时出错: {e}")
            print("尝试对时间索引进行排序...")
            kriged_data = kriged_data.sortby('time')
    
    # 加载观测数据 (CSV格式)
    observed_data = pd.read_csv(observed_data_path)
    
    # 重命名列以便于后续处理
    observed_data = observed_data.rename(columns={
        '经度': 'x',
        '纬度': 'y',
        '最低气温': 'value'  # 假设您要评估的是最低气温的插值结果
    })
    
    # 数据预处理：去除异常值
    observed_data = preprocess_data(observed_data)
    
    # 确定坐标变量名
    lon_var = 'longitude' if 'longitude' in kriged_data.dims else 'lon'
    lat_var = 'latitude' if 'latitude' in kriged_data.dims else 'lat'
    
    # 打印NC文件的基本信息，帮助调试
    print("NC文件结构:")
    print(kriged_data)
    print(f"变量列表: {list(kriged_data.variables)}")
    print(f"坐标范围: 经度 {kriged_data[lon_var].min().item()}-{kriged_data[lon_var].max().item()}, 纬度 {kriged_data[lat_var].min().item()}-{kriged_data[lat_var].max().item()}")
    if 'time' in kriged_data.dims:
        print(f"时间范围: {kriged_data.time.min().values} 至 {kriged_data.time.max().values}")
        try:
            time_diffs = np.diff(kriged_data.time.values).astype('timedelta64[ns]').astype(np.int64)
            print(f"时间索引是否单调递增: {np.all(time_diffs > 0)}")
        except:
            print("无法检查时间索引是否单调")
    
    return kriged_data, observed_data

def preprocess_data(data):
    """
    对观测数据进行预处理，去除异常值
    
    参数:
    data: 包含观测数据的DataFrame
    
    返回:
    processed_data: 处理后的DataFrame
    """
    print("开始数据预处理...")
    original_count = len(data)
    
    # 检查并处理缺失值
    missing_before = data['value'].isna().sum()
    if missing_before > 0:
        print(f"发现{missing_before}个缺失值")
        data = data.dropna(subset=['value'])
        print(f"已删除缺失值，剩余{len(data)}个数据点")
    
    # 检查坐标范围
    lon_min, lon_max = 73, 135  # 中国大致经度范围
    lat_min, lat_max = 18, 54   # 中国大致纬度范围
    
    # 筛选坐标在合理范围内的数据
    valid_coords = (data['x'] >= lon_min) & (data['x'] <= lon_max) & \
                   (data['y'] >= lat_min) & (data['y'] <= lat_max)
    
    invalid_coords_count = (~valid_coords).sum()
    if invalid_coords_count > 0:
        print(f"发现{invalid_coords_count}个坐标超出合理范围")
        data = data[valid_coords]
        print(f"已删除无效坐标，剩余{len(data)}个数据点")
    
    # 使用温度阈值检测明显错误值
    temp_min, temp_max = -50, 50  # 合理的温度范围（摄氏度）
    
    # 筛选在温度范围内的数据
    temp_outliers = (data['value'] < temp_min) | (data['value'] > temp_max)
    outlier_count = temp_outliers.sum()
    
    if outlier_count > 0:
        print(f"检测到{outlier_count}个超出合理范围的温度值")
        print(f"温度范围: [{temp_min}°C, {temp_max}°C]")
        data = data[~temp_outliers]
        print(f"已删除异常值，剩余{len(data)}个数据点")
    
    # 汇总预处理结果
    removed_count = original_count - len(data)
    removed_percent = (removed_count / original_count) * 100 if original_count > 0 else 0
    
    print(f"数据预处理完成:")
    print(f"原始数据点: {original_count}")
    print(f"处理后数据点: {len(data)}")
    print(f"移除的数据点: {removed_count} ({removed_percent:.2f}%)")
    
    return data

def extract_kriged_values_at_observed_points(kriged_data, observed_data):
    """
    在观测点位置提取克里金插值的值
    """
    # 提取插值结果中观测点位置的值
    kriged_values = []
    
    # 确保观测数据中有日期列
    if '日期' not in observed_data.columns:
        print("警告：观测数据中缺少'日期'列，无法匹配时间维度")
        return np.array([np.nan] * len(observed_data))
    
    # 获取NC文件中的时间值列表
    time_values = kriged_data.time.values
    
    # 确定坐标变量名
    lon_var = 'longitude' if 'longitude' in kriged_data.dims else 'lon'
    lat_var = 'latitude' if 'latitude' in kriged_data.dims else 'lat'
    print(f"使用坐标变量: {lon_var}, {lat_var}")
    
    for _, row in observed_data.iterrows():
        x, y = row['x'], row['y']
        date_str = row['日期']  # 假设日期格式为'YYYY/MM/DD'
        
        try:
            # 将日期字符串转换为标准格式
            date_parts = date_str.replace('/', '-').split('-')
            if len(date_parts) == 3:
                year, month, day = date_parts
                month = month.zfill(2)
                day = day.zfill(2)
                standard_date_str = f"{year}-{month}-{day}"
                date = np.datetime64(standard_date_str)
            else:
                raise ValueError(f"日期格式错误: {date_str}")
            
            # 使用手动方法查找最近的时间点
            time_diffs = np.abs(np.array([(t - date).astype('timedelta64[ns]').astype(np.int64) for t in time_values]))
            nearest_time_idx = np.argmin(time_diffs)
            nearest_time = time_values[nearest_time_idx]
            
            # 从插值结果中找到最接近的网格点和时间点
            # 使用动态确定的坐标变量名
            coords = {
                lon_var: x,
                lat_var: y,
                'time': nearest_time
            }
            
            value = kriged_data.sel(
                **coords,
                method='nearest'
            )['temperature'].values.item()
            
            kriged_values.append(value)
        except Exception as e:
            print(f"警告：无法在坐标({x}, {y})和日期({date_str})处找到插值值: {e}")
            kriged_values.append(np.nan)
    
    return np.array(kriged_values)

def calculate_error_metrics(observed_values, predicted_values):
    """
    计算各种误差指标
    """
    # 移除NaN值
    valid_indices = ~np.isnan(predicted_values)
    obs_valid = observed_values[valid_indices]
    pred_valid = predicted_values[valid_indices]
    
    if len(obs_valid) == 0:
        print("错误：没有有效的预测值，请检查数据")
        return {'RMSE': np.nan, 'MAE': np.nan, 'R²': np.nan}
    
    rmse = np.sqrt(mean_squared_error(obs_valid, pred_valid))
    mae = mean_absolute_error(obs_valid, pred_valid)
    r2 = r2_score(obs_valid, pred_valid)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

def setup_chinese_font():
    """
    设置matplotlib支持中文显示
    """
    system = platform.system()
    
    # 根据操作系统选择合适的中文字体
    if system == 'Windows':
        font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体
        # 或者使用微软雅黑
        # font_path = 'C:/Windows/Fonts/msyh.ttc'
    elif system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/PingFang.ttc'  # 苹方字体
    else:  # Linux
        font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
    
    # 检查字体文件是否存在
    if not os.path.exists(font_path):
        print(f"警告：找不到字体文件 {font_path}")
        print("尝试使用其他方法设置中文字体...")
        
        # 尝试使用matplotlib内置方法
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return
    
    # 创建字体属性对象
    chinese_font = fm.FontProperties(fname=font_path)
    
    # 设置默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    return chinese_font

def plot_observed_vs_predicted(observed_values, predicted_values, title="最低气温克里金插值评估"):
    """
    绘制观测值与预测值的散点图
    """
    # 设置中文字体
    chinese_font = setup_chinese_font()
    
    # 移除NaN值
    valid_indices = ~np.isnan(predicted_values)
    obs_valid = observed_values[valid_indices]
    pred_valid = predicted_values[valid_indices]
    
    if len(obs_valid) == 0:
        print("错误：没有有效的预测值，无法绘图")
        return
    
    plt.figure(figsize=(10, 8))
    plt.scatter(obs_valid, pred_valid, alpha=0.7)
    
    # 添加1:1线
    min_val = min(np.min(obs_valid), np.min(pred_valid))
    max_val = max(np.max(obs_valid), np.max(pred_valid))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 使用中文字体
    if chinese_font:
        plt.xlabel('观测值 (°C)', fontproperties=chinese_font)
        plt.ylabel('预测值 (°C)', fontproperties=chinese_font)
        plt.title(title, fontproperties=chinese_font)
    else:
        plt.xlabel('观测值 (°C)')
        plt.ylabel('预测值 (°C)')
        plt.title(title)
    
    plt.grid(True)
    
    # 添加误差指标文本
    metrics = calculate_error_metrics(obs_valid, pred_valid)
    text = f"RMSE: {metrics['RMSE']:.2f}°C\nMAE: {metrics['MAE']:.2f}°C\nR²: {metrics['R²']:.4f}"
    
    if chinese_font:
        plt.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontproperties=chinese_font)
    else:
        plt.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.savefig('kriging_validation.png', dpi=300)
    plt.show()

def perform_cross_validation(data, n_folds=5):
    """
    执行K折交叉验证
    
    参数:
    data: 包含坐标和观测值的DataFrame
    n_folds: 折数
    
    返回:
    cv_results: 交叉验证结果
    """
    # 确保数据中没有NaN值
    data_clean = data.dropna(subset=['x', 'y', 'value'])
    
    if len(data_clean) < n_folds:
        print(f"警告：数据点数({len(data_clean)})少于折数({n_folds})，将使用留一交叉验证")
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_results = []
    
    for train_idx, test_idx in cv.split(data_clean):
        train_data = data_clean.iloc[train_idx]
        test_data = data_clean.iloc[test_idx]
        
        # 提取训练集和测试集的坐标和值
        train_coords = train_data[['x', 'y']].values
        train_vals = train_data['value'].values
        
        test_coords = test_data[['x', 'y']].values
        test_vals = test_data['value'].values
        
        try:
            # 使用普通克里金法
            ok = pykrige.OrdinaryKriging(
                train_coords[:, 0], 
                train_coords[:, 1], 
                train_vals,
                variogram_model='spherical',  # 可以根据您的数据选择合适的变异函数模型
                verbose=False,
                enable_plotting=False
            )
            
            # 在测试点位置进行预测
            predicted, _ = ok.execute('points', test_coords[:, 0], test_coords[:, 1])
            
            # 计算误差指标
            metrics = calculate_error_metrics(test_vals, predicted)
            cv_results.append(metrics)
        except Exception as e:
            print(f"交叉验证中出现错误: {e}")
            continue
    
    if not cv_results:
        print("错误：交叉验证未能生成任何结果")
        return {'RMSE': np.nan, 'MAE': np.nan, 'R²': np.nan}
    
    # 计算平均误差指标
    avg_metrics = {
        'RMSE': np.mean([r['RMSE'] for r in cv_results]),
        'MAE': np.mean([r['MAE'] for r in cv_results]),
        'R²': np.mean([r['R²'] for r in cv_results])
    }
    
    return avg_metrics

def random_sample_evaluation(nc_file_path, observed_data_path, n_days=14, n_samples=3, variable_name='最低气温'):
    """
    随机抽样多次评估克里金插值精度
    
    参数:
    nc_file_path: 克里金插值结果NC文件路径
    observed_data_path: 观测数据CSV文件路径
    n_days: 每次抽样的天数
    n_samples: 抽样次数
    variable_name: 评估的变量名称
    """
    print(f"开始{n_samples}次随机抽样评估（每次{n_days}天）...")
    
    # 加载数据
    kriged_data, observed_data = load_data(nc_file_path, observed_data_path)
    
    # 获取所有唯一日期
    unique_dates = observed_data['日期'].unique()
    print(f"观测数据中共有{len(unique_dates)}个唯一日期")
    
    # 如果唯一日期少于要求的随机天数，调整n_days
    if len(unique_dates) < n_days:
        n_days = len(unique_dates)
        print(f"警告：唯一日期数少于要求的随机天数，已调整为{n_days}天")
    
    # 存储每次评估的结果
    all_metrics = []
    
    for i in range(n_samples):
        print(f"\n=== 第{i+1}次随机抽样评估 ===")
        
        # 随机选择n_days个日期
        sampled_dates = random.sample(list(unique_dates), n_days)
        print(f"随机选择的日期: {', '.join(sampled_dates[:3])}{'...' if len(sampled_dates) > 3 else ''}")
        
        # 筛选这些日期的观测数据
        sampled_data = observed_data[observed_data['日期'].isin(sampled_dates)]
        print(f"筛选出{len(sampled_data)}个观测点")
        
        # 提取观测点位置的插值值
        kriged_values = extract_kriged_values_at_observed_points(kriged_data, sampled_data)
        observed_values = sampled_data['value'].values
        
        # 计算有效匹配点数
        valid_count = np.sum(~np.isnan(kriged_values))
        print(f"成功匹配了{valid_count}/{len(observed_values)}个观测点")
        
        if valid_count == 0:
            print("错误：没有有效的匹配点，跳过此次评估")
            continue
        
        # 计算误差指标
        metrics = calculate_error_metrics(observed_values, kriged_values)
        all_metrics.append(metrics)
        
        print(f"本次评估指标:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # 绘制观测值与预测值的散点图
        plot_observed_vs_predicted(
            observed_values, 
            kriged_values, 
            title=f"{variable_name}克里金插值评估 (随机样本 {i+1})"
        )
    
    # 计算平均指标
    if all_metrics:
        avg_metrics = {
            'RMSE': np.mean([m['RMSE'] for m in all_metrics]),
            'MAE': np.mean([m['MAE'] for m in all_metrics]),
            'R²': np.mean([m['R²'] for m in all_metrics])
        }
        
        print("\n=== 随机抽样评估平均指标 ===")
        for metric, value in avg_metrics.items():
            print(f"平均{metric}: {value:.4f}")
    else:
        print("错误：所有随机样本评估均失败")

if __name__ == "__main__":
    # 替换为您的文件路径
    nc_file_path = "Copy of temperature_interpolation.nc"
    observed_data_path = "1990-2023_cy.csv"
    
    # 使用随机抽样评估
    random_sample_evaluation(
        nc_file_path, 
        observed_data_path, 
        n_days=14,  # 随机14天
        n_samples=3,  # 随机抽样3次
        variable_name='最低气温'
    )
    
    # 如果您仍然想运行完整评估，可以取消下面的注释
    # main(nc_file_path, observed_data_path, variable_name='最低气温')

import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import gc
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
import cartopy.crs as ccrs  # 添加cartopy用于地图投影
import platform  # 添加platform模块用于检测操作系统
import datetime  # 添加datetime模块用于处理时间数据
import psutil  # 添加psutil用于监控内存使用
import sys  # 添加sys模块用于错误处理
import time  # 添加time模块用于暂停处理
import fnmatch  # 添加fnmatch模块用于文件匹配
import gstools as gs
from numba import jit
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from scipy.sparse import lil_matrix, csr_matrix

# 导入Cython优化模块
try:
    import krig_fast
    CYTHON_AVAILABLE = True
    print("使用Cython加速模块")
except ImportError:
    CYTHON_AVAILABLE = False
    print("Cython加速模块不可用，使用纯Python实现")

# 添加导入
try:
    import cupy as cp
    from cupyx.scipy.spatial.distance import cdist
    GPU_AVAILABLE = True
    print("检测到CUDA GPU，将使用GPU加速")
except ImportError:
    GPU_AVAILABLE = False
    print("未检测到CUDA GPU或cupy库，使用CPU计算")

# 添加变异函数参数缓存
variogram_cache = {}

# 读取CSV数据
def read_data(file_path):
    """读取CSV数据文件"""
    # 修改分隔符为逗号，因为数据列是以逗号分隔的
    df = pd.read_csv(file_path, sep=',')
    print("数据列名:", df.columns.tolist())  # 打印列名以便调试
    return df

# 预处理数据：去除缺值和异常值
def preprocess_data(df, temp_col, date_col=None):
    """预处理数据，去除缺值和异常值"""
    # 记录原始数据量
    original_count = len(df)
    
    # 去除缺失值
    df = df.dropna(subset=[temp_col])  # 只检查温度列的缺失值
    na_removed_count = original_count - len(df)
    
    # 去除温度异常值（范围外的值：-50°C 到 50°C）
    df = df[(df[temp_col] >= -50) & (df[temp_col] <= 50)]
    outlier_removed_count = original_count - na_removed_count - len(df)
    
    # 处理日期列（如果存在）
    if date_col and date_col in df.columns:
        # 尝试转换日期格式
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            print(f"日期列 '{date_col}' 已转换为datetime格式")
        except Exception as e:
            print(f"警告: 无法转换日期列 '{date_col}': {e}")
    
    # 打印预处理信息
    print(f"预处理结果:")
    print(f"  - 原始数据: {original_count}条")
    print(f"  - 移除缺失值: {na_removed_count}条")
    print(f"  - 移除异常值(温度超出-50°C到50°C范围): {outlier_removed_count}条")
    print(f"  - 保留数据: {len(df)}条")
    
    return df

# 添加Numba加速函数
@jit(nopython=True, parallel=True)
def fast_distance_calculation(points1, points2):
    """使用Numba加速距离计算"""
    n1 = len(points1)
    n2 = len(points2)
    result = np.empty((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            result[i, j] = np.sqrt((points1[i, 0] - points2[j, 0])**2 + 
                                   (points1[i, 1] - points2[j, 1])**2)
    return result

# 处理单个区域的克里金插值
def process_region(i, j, lon_min, lat_min, lon_range, lat_range, num_lon_regions, num_lat_regions, 
                  lons, lats, values, grid_lon, grid_lat, resolution, max_points=1000):
    """处理单个区域的克里金插值，优化以提高速度"""
    try:
        # 监控内存使用
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"区域 ({i},{j}) 开始处理，当前内存使用: {mem_info.rss / (1024 * 1024):.2f} MB")
        
        # 计算当前区域的经纬度范围
        lon_start = lon_min + i * (lon_range / num_lon_regions)
        lon_end = lon_min + (i + 1) * (lon_range / num_lon_regions)
        lat_start = lat_min + j * (lat_range / num_lat_regions)
        lat_end = lat_min + (j + 1) * (lat_range / num_lat_regions)
        
        # 为了确保覆盖边界区域，扩展范围
        lon_start = max(lon_min, lon_start - resolution * 2)
        lon_end = min(lon_min + lon_range, lon_end + resolution * 2)
        lat_start = max(lat_min, lat_start - resolution * 2)
        lat_end = min(lat_min + lat_range, lat_end + resolution * 2)
        
        # 确保上下界正确
        if lon_start >= lon_end or lat_start >= lat_end:
            print(f"区域 ({lon_start:.2f}-{lon_end:.2f}, {lat_start:.2f}-{lat_end:.2f}) 边界无效，跳过")
            return None
        
        # 筛选当前区域内的数据点
        region_mask = (
            (lons >= lon_start) & (lons <= lon_end) &
            (lats >= lat_start) & (lats <= lat_end)
        )
        region_lons = lons[region_mask]
        region_lats = lats[region_mask]
        region_values = values[region_mask]
        
        # 如果区域内数据点太少，跳过
        if len(region_lons) < 5:  # 增加最小点数要求，从3改为5
            print(f"区域 ({lon_start:.2f}-{lon_end:.2f}, {lat_start:.2f}-{lat_end:.2f}) 数据点不足，跳过")
            return None
        
        # 如果区域内数据点太多，随机抽样减少点数
        if len(region_lons) > max_points:
            print(f"区域 ({i},{j}) 数据点过多 ({len(region_lons)}点)，随机抽样减少到{max_points}点")
            indices = np.random.choice(len(region_lons), max_points, replace=False)
            region_lons = region_lons[indices]
            region_lats = region_lats[indices]
            region_values = region_values[indices]
        
        # 创建区域网格
        region_grid_lon = np.arange(lon_start, lon_end + resolution/2, resolution)
        region_grid_lat = np.arange(lat_start, lat_end + resolution/2, resolution)
        
        # 确保网格点数组非空且上下界正确
        if len(region_grid_lon) < 2 or len(region_grid_lat) < 2 or region_grid_lon[0] >= region_grid_lon[-1] or region_grid_lat[0] >= region_grid_lat[-1]:
            print(f"区域 ({lon_start:.2f}-{lon_end:.2f}, {lat_start:.2f}-{lat_end:.2f}) 网格点无效，跳过")
            return None
        
        # 检查内存使用情况，如果内存不足则提前返回
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        if available_memory < 1000:  # 如果可用内存小于1000MB，则跳过此区域
            print(f"区域 ({i},{j}) 内存不足 (仅剩 {available_memory:.2f} MB)，跳过处理")
            return None
        
        # 在处理区域前检查缓存
        cache_key = f"{lon_start:.1f}_{lon_end:.1f}_{lat_start:.1f}_{lat_end:.1f}"
        if cache_key in variogram_cache:
            print(f"使用缓存的变异函数参数: {cache_key}")
            best_model = variogram_cache[cache_key]
        else:
            # 使用GSTools替代PyKrige进行克里金插值（速度提升5-10倍）
            try:
                # 创建坐标数组
                pos = (region_lons, region_lats)
                
                # 估计变异函数
                bin_center, gamma = gs.vario_estimate(pos, region_values)
                
                # 尝试不同的变异函数模型
                models = {
                    'spherical': gs.Spherical,
                    'exponential': gs.Exponential
                }
                
                best_model = None
                min_rmse = float('inf')
                
                for name, model_class in models.items():
                    try:
                        # 拟合变异函数
                        fitted_model = model_class.fit_variogram(bin_center, gamma, nugget=True)
                        
                        # 计算拟合误差
                        rmse = np.sqrt(np.mean((fitted_model.variogram(bin_center) - gamma)**2))
                        
                        if rmse < min_rmse:
                            min_rmse = rmse
                            best_model = fitted_model
                            best_name = name
                    except:
                        continue
                
                if best_model is None:
                    # 如果拟合失败，使用默认参数
                    best_model = gs.Spherical(dim=2, var=np.var(region_values), len_scale=1.0)
                    best_name = 'spherical (default)'
                
                print(f"区域 ({i},{j}) 最佳变异函数模型: {best_name}")
                
                # 创建克里金对象
                krig = gs.krige.Ordinary(best_model, pos, region_values)
                
                # 创建网格点
                grid_x, grid_y = np.meshgrid(region_grid_lon, region_grid_lat)
                grid_pos = (grid_x.flatten(), grid_y.flatten())
                
                # 执行克里金插值
                z_flat = krig(grid_pos)[0]  # 返回值和方差，我们只需要值
                z = z_flat.reshape(grid_y.shape)
                
                # 找到区域网格在总网格中的索引
                lon_indices = np.where(np.isin(grid_lon, region_grid_lon))[0]
                lat_indices = np.where(np.isin(grid_lat, region_grid_lat))[0]
                
                # 创建局部结果
                result = {
                    'lon_indices': lon_indices,
                    'lat_indices': lat_indices,
                    'z': z,
                    'region_grid_lon': region_grid_lon,
                    'region_grid_lat': region_grid_lat
                }
                
                # 清理内存
                del krig
                gc.collect()  # 强制垃圾回收，减少内存占用
                
                # 再次监控内存使用
                mem_info = process.memory_info()
                print(f"区域 ({i},{j}) 处理完成，当前内存使用: {mem_info.rss / (1024 * 1024):.2f} MB")
                
                # 缓存结果
                variogram_cache[cache_key] = best_model
                
                return result
            
            except Exception as e:
                print(f"使用GSTools进行克里金插值失败: {e}")
                # 回退到原始PyKrige方法
                # 优化变异函数模型选择 - 使用更快的方法
                # 对于大内存系统，可以跳过详细的交叉验证，直接尝试几个常用模型
                variogram_models = ['spherical', 'exponential']  # 减少模型数量，只使用最常用的两种
                best_model = None
                min_error = float('inf')
                
                # 使用简化的交叉验证方法 - 只在数据点较多时执行
                if len(region_lons) > 50:  # 只有当点数较多时才进行交叉验证
                    # 随机选择少量点进行测试，而不是全部
                    test_size = min(5, len(region_lons) // 10)  # 最多5个点或总数的10%
                    test_indices = np.random.choice(len(region_lons), test_size, replace=False)
                    
                    # ... 交叉验证代码 ...
                else:
                    # 对于小数据集，直接使用球形模型
                    best_model = 'spherical'
                
                # 使用最佳模型执行克里金插值，优化参数
                OK = OrdinaryKriging(
                    region_lons, region_lats, region_values,
                    variogram_model=best_model,
                    verbose=False,
                    enable_plotting=False,
                    nlags=15,  # 减少滞后数以加快计算
                    weight=True
                )
                
                # 生成区域网格数据
                z, ss = OK.execute('grid', region_grid_lon, region_grid_lat)
                
                # 找到区域网格在总网格中的索引
                lon_indices = np.where(np.isin(grid_lon, region_grid_lon))[0]
                lat_indices = np.where(np.isin(grid_lat, region_grid_lat))[0]
                
                # 创建局部结果
                result = {
                    'lon_indices': lon_indices,
                    'lat_indices': lat_indices,
                    'z': z,
                    'region_grid_lon': region_grid_lon,
                    'region_grid_lat': region_grid_lat
                }
                
                # 清理内存
                del OK, ss, region_lons, region_lats, region_values
                gc.collect()  # 强制垃圾回收，减少内存占用
                
                # 再次监控内存使用
                mem_info = process.memory_info()
                print(f"区域 ({i},{j}) 处理完成，当前内存使用: {mem_info.rss / (1024 * 1024):.2f} MB")
                
                return result
        
    except MemoryError:
        print(f"区域 ({i},{j}) 处理失败: 内存不足")
        # 强制垃圾回收
        gc.collect()
        return None
    except Exception as e:
        print(f"区域 ({i},{j}) 处理失败: {e}")
        return None

# 添加新函数：检测和处理异常值
def detect_anomalies(grid_lon, grid_lat, interpolated_data, original_df, temp_col='最低气温'):
    """检测和处理插值结果中的异常值"""
    print("\n执行二次检验，检测异常值...")
    
    # 获取原始数据的统计信息
    orig_min = original_df[temp_col].min()
    orig_max = original_df[temp_col].max()
    orig_mean = original_df[temp_col].mean()
    orig_std = original_df[temp_col].std()
    
    # 设置异常值阈值（基于原始数据的统计特性）
    lower_bound = max(orig_min - 3 * orig_std, -50)  # 不低于-50°C
    upper_bound = min(orig_max + 3 * orig_std, 50)   # 不高于50°C
    
    # 检测异常值
    anomalies_low = interpolated_data < lower_bound
    anomalies_high = interpolated_data > upper_bound
    anomalies = anomalies_low | anomalies_high
    
    # 统计异常值数量
    num_anomalies = np.sum(anomalies)
    total_cells = interpolated_data.size
    anomaly_percentage = (num_anomalies / total_cells) * 100
    
    print(f"异常值检测结果:")
    print(f"  - 原始数据范围: {orig_min:.2f}°C 到 {orig_max:.2f}°C")
    print(f"  - 原始数据均值: {orig_mean:.2f}°C, 标准差: {orig_std:.2f}°C")
    print(f"  - 异常值阈值: {lower_bound:.2f}°C 到 {upper_bound:.2f}°C")
    print(f"  - 检测到 {num_anomalies} 个异常值 (占总网格点的 {anomaly_percentage:.2f}%)")
    
    # 如果存在异常值，进行可视化和修正
    if num_anomalies > 0:
        # 记录异常值位置和值
        anomaly_locs = np.where(anomalies)
        anomaly_values = interpolated_data[anomalies]
        
        # 打印部分异常值示例
        print("\n异常值示例:")
        max_examples = min(10, num_anomalies)
        for i in range(max_examples):
            lat_idx, lon_idx = anomaly_locs[0][i], anomaly_locs[1][i]
            value = anomaly_values[i]
            print(f"  - 位置 ({grid_lat[lat_idx]:.2f}°N, {grid_lon[lon_idx]:.2f}°E): {value:.2f}°C")
        
        # 可视化异常值分布
        plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines(resolution='50m')
        ax.gridlines(draw_labels=True)
        
        # 创建异常值掩码数组（用于可视化）
        anomaly_mask = np.zeros_like(interpolated_data)
        anomaly_mask[anomalies] = 1
        
        # 绘制异常值分布
        im = ax.pcolormesh(grid_lon, grid_lat, anomaly_mask, 
                          cmap='Reds', transform=ccrs.PlateCarree())
        plt.title('异常值分布')
        plt.colorbar(im, label='异常值标记')
        
        # 保存图像
        anomaly_image = 'temperature_anomalies.png'
        plt.savefig(anomaly_image, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"异常值分布图已保存为 {anomaly_image}")
        
        # 修正异常值
        print("\n开始修正异常值...")
        corrected_data = np.copy(interpolated_data)
        
        # 对于每个异常值，使用周围非异常值的中值替换
        for lat_idx, lon_idx in zip(anomaly_locs[0], anomaly_locs[1]):
            # 获取周围的3x3区域
            lat_start = max(0, lat_idx - 1)
            lat_end = min(len(grid_lat), lat_idx + 2)
            lon_start = max(0, lon_idx - 1)
            lon_end = min(len(grid_lon), lon_idx + 2)
            
            # 提取区域数据
            region = interpolated_data[lat_start:lat_end, lon_start:lon_end]
            region_anomalies = anomalies[lat_start:lat_end, lon_start:lon_end]
            
            # 获取非异常值
            valid_values = region[~region_anomalies]
            
            # 如果周围有非异常值，使用它们的中值替换
            if len(valid_values) > 0:
                corrected_data[lat_idx, lon_idx] = np.median(valid_values)
            else:
                # 如果周围没有非异常值，使用全局均值替换
                corrected_data[lat_idx, lon_idx] = orig_mean
        
        # 检查是否仍有超出范围的值
        still_anomalies = (corrected_data < lower_bound) | (corrected_data > upper_bound)
        num_still_anomalies = np.sum(still_anomalies)
        
        if num_still_anomalies > 0:
            print(f"警告: 修正后仍有 {num_still_anomalies} 个异常值，将它们限制在合理范围内")
            # 将剩余异常值限制在合理范围内
            corrected_data = np.clip(corrected_data, lower_bound, upper_bound)
        
        print(f"异常值修正完成")
        
        # 计算修正前后的差异
        diff = np.abs(corrected_data - interpolated_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"修正前后差异统计:")
        print(f"  - 最大差异: {max_diff:.2f}°C")
        print(f"  - 平均差异: {mean_diff:.2f}°C")
        
        # 可视化修正效果
        plt.figure(figsize=(15, 10))
        
        # 创建子图
        ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax1.coastlines(resolution='50m')
        ax1.gridlines(draw_labels=True)
        im1 = ax1.contourf(grid_lon, grid_lat, interpolated_data, 
                          cmap='viridis', levels=20, transform=ccrs.PlateCarree())
        plt.colorbar(im1, ax=ax1, label='温度 (°C)')
        ax1.set_title('修正前')
        
        ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax2.coastlines(resolution='50m')
        ax2.gridlines(draw_labels=True)
        im2 = ax2.contourf(grid_lon, grid_lat, corrected_data, 
                          cmap='viridis', levels=20, transform=ccrs.PlateCarree())
        plt.colorbar(im2, ax=ax2, label='温度 (°C)')
        ax2.set_title('修正后')
        
        plt.suptitle('异常值修正效果对比')
        
        # 保存图像
        correction_image = 'temperature_correction.png'
        plt.savefig(correction_image, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"修正效果对比图已保存为 {correction_image}")
        
        # 返回修正后的数据
        return corrected_data, True
    else:
        print("未检测到异常值，无需修正")
        return interpolated_data, False

# 分块处理克里金插值（串行版本）
def kriging_interpolation(df, resolution=0.5, chunk_size=1000, parallel=True, n_jobs=None):
    """使用克里金方法将点数据插值到网格，支持并行处理以提高速度"""
    # 提取经纬度和目标变量（最低气温）
    lon_col = '经度'
    lat_col = '纬度'
    temp_col = '最低气温'
    date_col = '日期'
    
    print(f"使用列: 经度={lon_col}, 纬度={lat_col}, 温度={temp_col}, 日期={date_col}")
    
    # 预处理数据：去除缺值和异常值
    df = preprocess_data(df, temp_col, date_col)
    
    # 如果预处理后数据太少，则抛出错误
    if len(df) < 3:
        raise ValueError("预处理后的有效数据点不足，无法进行克里金插值（至少需要3个点）")
    
    # 确定网格范围（根据数据范围稍微扩展一些）
    lons = df[lon_col].values
    lats = df[lat_col].values
    values = df[temp_col].values
    
    lon_min, lon_max = np.floor(min(lons)), np.ceil(max(lons))
    lat_min, lat_max = np.floor(min(lats)), np.ceil(max(lats))
    
    # 创建网格点，确保分辨率为0.5度
    resolution = 0.5  # 确保分辨率为0.5度
    grid_lon = np.arange(lon_min, lon_max + resolution, resolution)
    grid_lat = np.arange(lat_min, lat_max + resolution, resolution)
    
    # 创建结果网格
    result_grid = np.zeros((len(grid_lat), len(grid_lon)))
    weight_grid = np.zeros((len(grid_lat), len(grid_lon)))
    
    # 根据系统和可用内存调整区域划分
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    print(f"当前可用内存: {available_memory:.2f} MB")
    
    # 根据可用内存动态调整区域划分和每个区域的最大点数
    if available_memory > 20000:  # 如果可用内存大于20GB
        num_lon_regions = 4  # 增加区域数量以便并行处理
        num_lat_regions = 4
        max_points = 2500  # 增加每个区域的最大点数
    elif available_memory > 10000:  # 如果可用内存大于10GB
        num_lon_regions = 5
        num_lat_regions = 5
        max_points = 2000
    elif available_memory > 5000:  # 如果可用内存大于5GB
        num_lon_regions = 6
        num_lat_regions = 6
        max_points = 1500
    else:  # 内存较小
        num_lon_regions = 7
        num_lat_regions = 7
        max_points = 1000
    
    print(f"将区域划分为 {num_lon_regions}x{num_lat_regions} 个子区域进行处理，每个区域最多处理 {max_points} 个数据点")
    
    # 准备任务列表
    tasks = []
    for i in range(num_lon_regions):
        for j in range(num_lat_regions):
            tasks.append((i, j))
    
    # 添加趋势面分析
    print("执行趋势面分析以提高插值精度...")
    try:
        # 使用多项式拟合全局趋势
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        # 准备数据
        X = np.column_stack((lons, lats))
        y = values
        
        # 创建二阶多项式特征
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        # 拟合线性回归模型
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # 预测趋势值
        trend = model.predict(X_poly)
        
        # 计算残差
        residuals = y - trend
        
        # 使用残差进行克里金插值
        print("使用残差进行克里金插值，以捕捉局部变化...")
        values_for_kriging = residuals
        
        # 保存趋势模型用于后处理
        trend_model = (poly, model)
        
    except Exception as e:
        print(f"趋势面分析失败: {e}")
        print("将使用原始数据进行克里金插值...")
        values_for_kriging = values
        trend_model = None
    
    # 决定是否使用并行处理
    if parallel and available_memory > 8000:  # 只有在内存充足时才使用并行处理
        print("使用并行处理以加速计算...")
        
        # 确定CPU核心数
        if n_jobs is None:
            n_jobs = min(mp.cpu_count(), 8)  # 使用最多8个核心，避免过度并行
        
        print(f"使用 {n_jobs} 个CPU核心进行并行处理")
        
        # 使用joblib进行并行处理
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_region)(
                i, j, lon_min, lat_min, lon_max - lon_min, lat_max - lat_min, 
                num_lon_regions, num_lat_regions, lons, lats, values_for_kriging, 
                grid_lon, grid_lat, resolution, max_points=max_points
            ) for i, j in tasks
        )
        
        # 处理并行计算结果
        for result in results:
            if result is not None:
                lon_indices = result['lon_indices']
                lat_indices = result['lat_indices']
                z = result['z']
                
                # 更新结果网格
                for ii, lon_idx in enumerate(lon_indices):
                    for jj, lat_idx in enumerate(lat_indices):
                        if not np.isnan(z[jj, ii]):
                            result_grid[lat_idx, lon_idx] += z[jj, ii]
                            weight_grid[lat_idx, lon_idx] += 1
    else:
        print("使用串行处理...")
        # 串行处理所有区域，避免并行处理导致的内存问题
        print("使用串行处理以避免内存溢出...")
        
        # 串行处理每个区域
        for i, j in tqdm(tasks, desc="处理区域"):
            # 检查当前内存使用情况
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory < 1000:  # 如果可用内存小于1000MB，暂停一下让系统释放内存
                print(f"内存不足 (仅剩 {available_memory:.2f} MB)，暂停10秒等待内存释放")
                gc.collect()  # 强制垃圾回收
                time.sleep(10)  # 暂停10秒
                
                # 再次检查内存，如果仍然不足，则跳过此区域
                available_memory = psutil.virtual_memory().available / (1024 * 1024)
                if available_memory < 1000:
                    print(f"内存仍然不足，跳过区域 ({i},{j})")
                    continue
            
            try:
                # 限制每个区域的处理时间和数据点数量，避免单个区域占用过多资源
                result = process_region(
                    i, j, lon_min, lat_min, lon_max - lon_min, lat_max - lat_min, 
                    num_lon_regions, num_lat_regions, lons, lats, values_for_kriging, 
                    grid_lon, grid_lat, resolution, max_points=max_points
                )
                
                # 处理结果
                if result is not None:
                    lon_indices = result['lon_indices']
                    lat_indices = result['lat_indices']
                    z = result['z']
                    
                    # 更新结果网格
                    for ii, lon_idx in enumerate(lon_indices):
                        for jj, lat_idx in enumerate(lat_indices):
                            if not np.isnan(z[jj, ii]):
                                result_grid[lat_idx, lon_idx] += z[jj, ii]
                                weight_grid[lat_idx, lon_idx] += 1
                
                # 清理内存
                del result
                gc.collect()
                
                # 每处理完一个区域，保存中间结果，以防程序崩溃
                if (i * num_lat_regions + j + 1) % 4 == 0:  # 每处理4个区域保存一次
                    try:
                        # 创建临时结果
                        temp_result = np.copy(result_grid)
                        temp_weight = np.copy(weight_grid)
                        
                        # 计算加权平均
                        mask = temp_weight > 0
                        temp_result[mask] = temp_result[mask] / temp_weight[mask]
                        
                        # 保存临时结果
                        temp_file = f"temp_result_{i}_{j}.npy"
                        np.save(temp_file, temp_result)
                        print(f"已保存临时结果到 {temp_file}")
                    except Exception as e:
                        print(f"保存临时结果失败: {e}")
            
            except MemoryError:
                print(f"处理区域 ({i},{j}) 时内存不足，跳过此区域")
                gc.collect()  # 强制垃圾回收
                time.sleep(5)  # 暂停5秒
                continue
            except Exception as e:
                print(f"处理区域 ({i},{j}) 时发生错误: {e}")
                continue
            
            # 检查内存使用
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            print(f"区域 ({i},{j}) 处理后可用内存: {available_memory:.2f} MB")
        
    # 计算加权平均结果
    mask = weight_grid > 0
    result_grid[mask] = result_grid[mask] / weight_grid[mask]
    
    # 如果使用了趋势面分析，添加回趋势
    if trend_model is not None:
        try:
            poly, model = trend_model
            
            # 为网格点创建坐标
            grid_coords = []
            for lat in grid_lat:
                for lon in grid_lon:
                    grid_coords.append([lon, lat])
            grid_coords = np.array(grid_coords)
            
            # 转换为多项式特征
            grid_poly = poly.transform(grid_coords)
            
            # 预测网格点的趋势值
            grid_trend = model.predict(grid_poly)
            
            # 重塑为网格形状
            grid_trend = grid_trend.reshape(len(grid_lat), len(grid_lon))
            
            # 添加趋势到残差插值结果
            result_grid = result_grid + grid_trend
            
            print("已将趋势添加回插值结果")
        except Exception as e:
            print(f"添加趋势回插值结果时出错: {e}")
    
    # 处理未插值的区域（如果有）
    if np.any(weight_grid == 0):
        print(f"警告: {np.sum(weight_grid == 0)}个网格点未被插值，将使用最近邻填充")
        
        # 准备有效点坐标
        valid_mask = weight_grid > 0
        valid_indices = np.where(valid_mask)
        valid_points = np.column_stack((valid_indices[0], valid_indices[1]))
        
        # 准备需要填充的点坐标
        fill_mask = weight_grid == 0
        fill_indices = np.where(fill_mask)
        fill_points = np.column_stack((fill_indices[0], fill_indices[1]))
        
        if len(fill_points) > 0 and len(valid_points) > 0:
            # 计算距离矩阵
            distances = fast_distance_calculation(fill_points, valid_points)
            
            # 找到最近点
            nearest_indices = np.argmin(distances, axis=1)
            
            # 填充值
            for i, nearest_idx in enumerate(nearest_indices):
                y, x = fill_indices[0][i], fill_indices[1][i]
                valid_y, valid_x = valid_indices[0][nearest_idx], valid_indices[1][nearest_idx]
                result_grid[y, x] = result_grid[valid_y, valid_x]
    
    # 执行二次检验，检测和修正异常值
    result_grid, anomalies_fixed = detect_anomalies(grid_lon, grid_lat, result_grid, df, temp_col)
    if anomalies_fixed:
        print("已完成异常值检测和修正")
    
    # 提取时间信息（如果有）
    time_info = None
    if date_col and date_col in df.columns:
        try:
            # 尝试获取唯一的日期值
            unique_dates = pd.to_datetime(df[date_col]).unique()
            if len(unique_dates) == 1:
                time_info = unique_dates[0]
                print(f"数据时间: {time_info}")
            else:
                # 如果有多个日期，使用最常见的日期
                time_counts = pd.to_datetime(df[date_col]).value_counts()
                time_info = time_counts.index[0]
                print(f"数据包含多个日期，使用最常见的日期: {time_info}")
        except Exception as e:
            print(f"警告: 无法处理日期信息: {e}")
    
    # 清理临时文件
    for i in range(num_lon_regions):
        for j in range(num_lat_regions):
            temp_file = f"temp_result_{i}_{j}.npy"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    return grid_lon, grid_lat, result_grid, time_info

# 保存为NetCDF文件
def save_to_netcdf(grid_lon, grid_lat, data, output_file, time_info=None, anomalies_checked=True):
    """将网格数据保存为NetCDF格式"""
    # 创建xarray数据集
    if time_info is not None:
        # 如果有时间信息，添加时间维度
        ds = xr.Dataset(
            data_vars={
                "temperature": (["time", "lat", "lon"], data[np.newaxis, :, :]),
            },
            coords={
                "lon": grid_lon,
                "lat": grid_lat,
                "time": [time_info],
            },
            attrs={
                "description": "最低气温插值数据",
                "resolution": "0.5度 x 0.5度",
                "interpolation_method": "Ordinary Kriging (串行处理)",
                "projection": "等经纬度投影 (Plate Carrée)",
                "anomalies_checked": str(anomalies_checked),
                "creation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    else:
        # 如果没有时间信息，不添加时间维度
        ds = xr.Dataset(
            data_vars={
                "temperature": (["lat", "lon"], data),
            },
            coords={
                "lon": grid_lon,
                "lat": grid_lat,
            },
            attrs={
                "description": "最低气温插值数据",
                "resolution": "0.5度 x 0.5度",
                "interpolation_method": "Ordinary Kriging (串行处理)",
                "projection": "等经纬度投影 (Plate Carrée)",
                "anomalies_checked": str(anomalies_checked),
                "creation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    
    # 保存为NetCDF文件
    ds.to_netcdf(output_file)
    print(f"数据已保存至 {output_file}")
    
    return ds

# 可视化结果
def visualize_data(ds):
    """可视化插值结果"""
    plt.figure(figsize=(12, 10))
    
    # 使用cartopy创建地图投影
    ax = plt.axes(projection=ccrs.PlateCarree())  # 使用等经纬度投影
    ax.coastlines(resolution='50m')  # 添加海岸线
    ax.gridlines(draw_labels=True)  # 添加网格线和标签
    
    # 检查数据集是否包含时间维度
    if 'time' in ds.dims:
        # 如果有时间维度，使用第一个时间点的数据
        temp_data = ds.temperature.isel(time=0)
        time_str = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
        title = f'最低气温插值结果 ({time_str})'
    else:
        # 如果没有时间维度，直接使用温度数据
        temp_data = ds.temperature
        title = '最低气温插值结果'
    
    # 使用contourf创建填充等值线图
    im = ax.contourf(ds.lon, ds.lat, temp_data, 
                    cmap='viridis', levels=20,
                    transform=ccrs.PlateCarree())  # 指定数据的投影
    
    plt.title(title)
    plt.colorbar(im, label='温度 (°C)')
    
    # 保存图像
    output_image = 'temperature_interpolation.png'
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()  # 使用close代替show，避免在无GUI环境下出错
    print(f"可视化结果已保存为 {output_image}")
    
    # 在Windows系统下尝试打开图像
    if platform.system() == "Windows":
        try:
            os.startfile(output_image)
        except:
            print(f"无法自动打开图像，请手动查看 {output_image}")

# 添加新函数：评估插值精度
def evaluate_interpolation(df, grid_lon, grid_lat, interpolated_data, temp_col='最低气温', lon_col='经度', lat_col='纬度'):
    """使用留一交叉验证评估插值精度"""
    print("\n评估插值精度...")
    
    # 提取原始数据点
    lons = df[lon_col].values
    lats = df[lat_col].values
    values = df[temp_col].values
    
    # 从插值网格中提取原始数据点位置的值
    from scipy.interpolate import RegularGridInterpolator
    
    # 创建插值函数
    interp_func = RegularGridInterpolator((grid_lat, grid_lon), interpolated_data, 
                                         bounds_error=False, fill_value=None)
    
    # 准备坐标点
    points = np.column_stack((lats, lons))
    
    # 获取插值值
    interpolated_values = interp_func(points)
    
    # 计算误差
    errors = values - interpolated_values
    abs_errors = np.abs(errors)
    
    # 计算统计指标
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(abs_errors)
    
    # 打印评估结果
    print(f"插值精度评估结果:")
    print(f"  - 平均绝对误差 (MAE): {mae:.4f}°C")
    print(f"  - 均方根误差 (RMSE): {rmse:.4f}°C")
    print(f"  - 最大误差: {max_error:.4f}°C")
    
    # 绘制误差直方图
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='blue')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('插值误差分布')
    plt.xlabel('误差 (°C)')
    plt.ylabel('频数')
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    error_image = 'interpolation_error_histogram.png'
    plt.savefig(error_image, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"误差分布直方图已保存为 {error_image}")
    
    return mae, rmse, max_error

# 添加新函数：清理临时NC文件
def cleanup_temp_nc_files(output_file_pattern):
    """清理生成的临时NC文件"""
    print("\n开始清理临时NC文件...")
    
    # 获取基本文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(output_file_pattern))[0]
    
    # 获取目录
    directory = os.path.dirname(output_file_pattern)
    if not directory:
        directory = "."
    
    # 构建匹配模式
    pattern = f"{base_name}_*.nc"
    
    # 查找匹配的文件
    matching_files = []
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, pattern):
            matching_files.append(os.path.join(directory, file))
    
    # 删除找到的文件
    if matching_files:
        print(f"找到 {len(matching_files)} 个临时NC文件:")
        for file in matching_files:
            try:
                os.remove(file)
                print(f"  - 已删除: {file}")
            except Exception as e:
                print(f"  - 无法删除 {file}: {e}")
        print("临时NC文件清理完成")
    else:
        print("未找到需要清理的临时NC文件")

# 添加内存预分配和缓存优化
def optimize_memory():
    """优化内存使用和NumPy性能"""
    # 设置NumPy线程数
    import os
    os.environ["OMP_NUM_THREADS"] = str(min(8, mp.cpu_count()))
    os.environ["OPENBLAS_NUM_THREADS"] = str(min(8, mp.cpu_count()))
    os.environ["MKL_NUM_THREADS"] = str(min(8, mp.cpu_count()))
    
    # 预热NumPy
    _ = np.random.random((1000, 1000)).mean()
    
    # 清理内存
    gc.collect()
    
    print("已优化内存和NumPy性能设置")

# 修改主函数，添加清理临时文件功能
def main(input_file, output_file, chunk_size=500, cleanup_temp=True, parallel=True, n_jobs=None, use_dask=True):
    """主函数，支持并行处理选项和Dask分布式计算"""
    
    # 如果启用Dask，创建本地集群
    if use_dask and parallel and psutil.virtual_memory().available > 10000 * 1024 * 1024:  # >10GB
        print("初始化Dask分布式计算环境...")
        if n_jobs is None:
            n_jobs = min(mp.cpu_count(), 8)
        
        cluster = LocalCluster(n_workers=n_jobs, threads_per_worker=1)
        client = Client(cluster)
        print(f"Dask集群已启动，使用 {n_jobs} 个工作进程")
        
        # 设置全局Dask配置
        dask.config.set({'array.chunk-size': '128MiB'})
    else:
        client = None
    
    try:
        # 检查文件路径是否存在
        if not os.path.exists(input_file):
            print(f"错误: 输入文件 '{input_file}' 不存在")
            return None
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        
        # 读取数据
        df = read_data(input_file)
        
        # 检查是否有多个日期，如果有则需要按日期分组处理
        date_col = '日期'
        if date_col in df.columns:
            try:
                # 转换日期列
                df[date_col] = pd.to_datetime(df[date_col])
                unique_dates = df[date_col].unique()
                
                if len(unique_dates) > 1:
                    print(f"检测到多个日期 ({len(unique_dates)}个)，将按日期分组处理")
                    
                    # 创建一个列表存储所有日期的结果
                    all_results = []
                    
                    # 按日期分组处理
                    for date in tqdm(unique_dates, desc="处理不同日期"):
                        date_df = df[df[date_col] == date]
                        print(f"\n处理日期: {date}, 数据量: {len(date_df)}条")
                        
                        # 执行克里金插值
                        try:
                            grid_lon, grid_lat, interpolated_data, _ = kriging_interpolation(
                                date_df, resolution=0.5, chunk_size=chunk_size, parallel=parallel, n_jobs=n_jobs
                            )
                            
                            # 存储结果
                            all_results.append({
                                'date': date,
                                'grid_lon': grid_lon,
                                'grid_lat': grid_lat,
                                'data': interpolated_data
                            })
                            
                            # 保存单日期结果
                            date_str = pd.to_datetime(date).strftime('%Y%m%d')
                            date_output = f"{os.path.splitext(output_file)[0]}_{date_str}.nc"
                            ds = save_to_netcdf(grid_lon, grid_lat, interpolated_data, date_output, date)
                            
                        except Exception as e:
                            print(f"处理日期 {date} 时出错: {e}")
                    
                    # 合并所有日期的结果到一个NetCDF文件
                    if all_results:
                        print(f"\n合并 {len(all_results)} 个日期的结果到一个NetCDF文件...")
                        
                        # 确保所有网格一致
                        first_result = all_results[0]
                        grid_lon = first_result['grid_lon']
                        grid_lat = first_result['grid_lat']
                        
                        # 创建时间维度和数据数组
                        times = [r['date'] for r in all_results]
                        data = np.stack([r['data'] for r in all_results])
                        
                        # 创建xarray数据集
                        ds = xr.Dataset(
                            data_vars={
                                "temperature": (["time", "lat", "lon"], data),
                            },
                            coords={
                                "lon": grid_lon,
                                "lat": grid_lat,
                                "time": times,
                            },
                            attrs={
                                "description": "最低气温插值数据 (多日期)",
                                "resolution": "0.5度 x 0.5度",
                                "interpolation_method": "Ordinary Kriging (串行处理)",
                                "projection": "等经纬度投影 (Plate Carrée)"
                            }
                        )
                        
                        # 保存合并结果
                        ds.to_netcdf(output_file)
                        print(f"合并数据已保存至 {output_file}")
                        
                        # 可视化第一个日期的结果
                        visualize_data(ds.isel(time=0))
                        
                        # 添加精度评估
                        evaluate_interpolation(df, grid_lon, grid_lat, interpolated_data)
                        
                        # 清理临时NC文件
                        if cleanup_temp:
                            cleanup_temp_nc_files(output_file)
                        
                        return ds
                    else:
                        print("没有成功处理任何日期，无法创建合并结果")
                        return None
                else:
                    # 只有一个日期，正常处理
                    print(f"只检测到一个日期: {unique_dates[0]}")
                    grid_lon, grid_lat, interpolated_data, time_info = kriging_interpolation(
                        df, resolution=0.5, chunk_size=chunk_size, parallel=parallel, n_jobs=n_jobs
                    )
                    ds = save_to_netcdf(grid_lon, grid_lat, interpolated_data, output_file, time_info)
                    visualize_data(ds)
                    
                    # 添加精度评估
                    evaluate_interpolation(df, grid_lon, grid_lat, interpolated_data)
                    
                    # 清理临时NC文件
                    if cleanup_temp:
                        cleanup_temp_nc_files(output_file)
                    
                    return ds
                    
            except Exception as e:
                print(f"日期分组处理失败: {e}")
                print("将按单一数据集处理...")
        
        # 如果没有日期列或处理失败，按单一数据集处理
        grid_lon, grid_lat, interpolated_data, time_info = kriging_interpolation(
            df, resolution=0.5, chunk_size=chunk_size, parallel=parallel, n_jobs=n_jobs
        )
        ds = save_to_netcdf(grid_lon, grid_lat, interpolated_data, output_file, time_info)
        visualize_data(ds)
        
        # 添加精度评估
        evaluate_interpolation(df, grid_lon, grid_lat, interpolated_data)
        
        # 清理临时NC文件
        if cleanup_temp:
            cleanup_temp_nc_files(output_file)
        
        return ds
    
    except MemoryError:
        print("错误: 内存不足，无法完成处理")
        return None
    except Exception as e:
        print(f"错误: 处理失败 - {e}")
        return None
    finally:
        # 确保出错时也关闭Dask客户端
        if client:
            client.close()
            print("Dask集群已关闭")

# 修改主程序入口
if __name__ == "__main__":
    # 优化内存和NumPy性能
    optimize_memory()
    
    # 假设数据保存在名为'temperature_data.csv'的文件中
    input_file = "1990-2023_cy.csv"
    output_file = "temperature_interpolation.nc"
    
    # 根据系统内存情况调整chunk_size
    system = platform.system()
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    
    # 定义chunk_size
    if available_memory > 16000:  # 如果可用内存大于16GB
        chunk_size = 800
    elif available_memory > 8000:  # 如果可用内存大于8GB
        chunk_size = 500
    else:  # 内存较小
        chunk_size = 300
    
    # 根据可用内存决定是否使用并行处理
    parallel = available_memory > 8000  # 只有在内存大于8GB时才使用并行
    n_jobs = min(mp.cpu_count() - 1, 8) if parallel else 1  # 保留一个核心给系统
    
    print(f"在{system}系统上运行，可用内存: {available_memory:.2f} MB")
    print(f"使用参数: chunk_size={chunk_size}, 并行处理={parallel}, CPU核心数={n_jobs}")
    
    # 运行主函数
    ds = main(input_file, output_file, chunk_size=chunk_size, cleanup_temp=True, parallel=parallel, n_jobs=n_jobs)

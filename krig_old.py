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

# 处理单个区域的克里金插值
def process_region(i, j, lon_min, lat_min, lon_range, lat_range, num_lon_regions, num_lat_regions, 
                  lons, lats, values, grid_lon, grid_lat, resolution, max_points=1000):
    """处理单个区域的克里金插值，用于并行计算，限制最大处理点数"""
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
        if len(region_lons) < 3:
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
        
        # 执行克里金插值
        OK = OrdinaryKriging(
            region_lons, region_lats, region_values,
            variogram_model='spherical',
            verbose=False,
            enable_plotting=False
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

# 分块处理克里金插值（串行版本）
def kriging_interpolation(df, resolution=0.5, chunk_size=1000):
    """使用克里金方法将点数据插值到网格，采用分块处理减少内存使用"""
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
    
    # 将数据分成多个区域进行处理
    print(f"将数据分成多个区域进行克里金插值，以减少内存使用...")
    
    # 计算经纬度范围并划分区域
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    
    # 根据操作系统调整区域数量
    system = platform.system()
    print(f"检测到操作系统: {system}")
    
    # 根据系统和可用内存调整区域划分
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    print(f"当前可用内存: {available_memory:.2f} MB")
    
    # 根据可用内存动态调整区域划分和每个区域的最大点数
    if available_memory > 20000:  # 如果可用内存大于20GB
        num_lon_regions = 3
        num_lat_regions = 3
        max_points = 2000
    elif available_memory > 10000:  # 如果可用内存大于10GB
        num_lon_regions = 4
        num_lat_regions = 4
        max_points = 1500
    elif available_memory > 5000:  # 如果可用内存大于5GB
        num_lon_regions = 5
        num_lat_regions = 5
        max_points = 1000
    else:  # 内存较小
        num_lon_regions = 6
        num_lat_regions = 6
        max_points = 500
    
    print(f"将区域划分为 {num_lon_regions}x{num_lat_regions} 个子区域进行处理，每个区域最多处理 {max_points} 个数据点")
    
    # 串行处理所有区域，避免并行处理导致的内存问题
    print("使用串行处理以避免内存溢出...")
    
    # 准备任务列表
    tasks = []
    for i in range(num_lon_regions):
        for j in range(num_lat_regions):
            tasks.append((i, j))
    
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
                i, j, lon_min, lat_min, lon_range, lat_range, 
                num_lon_regions, num_lat_regions, lons, lats, values, 
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
    
    # 处理未插值的区域（如果有）
    if np.any(weight_grid == 0):
        print(f"警告: {np.sum(weight_grid == 0)}个网格点未被插值，将使用最近邻填充")
        # 使用最近邻填充未插值的区域
        for i in range(len(grid_lat)):
            for j in range(len(grid_lon)):
                if weight_grid[i, j] == 0:
                    # 找到最近的有值的网格点
                    valid_mask = weight_grid > 0
                    if not np.any(valid_mask):
                        continue
                    
                    # 计算到所有有效点的距离
                    y_indices, x_indices = np.where(valid_mask)
                    distances = np.sqrt((i - y_indices)**2 + (j - x_indices)**2)
                    
                    # 找到最近点
                    nearest_idx = np.argmin(distances)
                    result_grid[i, j] = result_grid[y_indices[nearest_idx], x_indices[nearest_idx]]
    
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
def save_to_netcdf(grid_lon, grid_lat, data, output_file, time_info=None):
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
                "projection": "等经纬度投影 (Plate Carrée)"
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
                "projection": "等经纬度投影 (Plate Carrée)"
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

# 主函数
def main(input_file, output_file, chunk_size=500):
    """主函数"""
    # 检查文件路径是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在")
        return None
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    try:
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
                                date_df, resolution=0.5, chunk_size=chunk_size
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
                        
                        return ds
                    else:
                        print("没有成功处理任何日期，无法创建合并结果")
                        return None
                else:
                    # 只有一个日期，正常处理
                    print(f"只检测到一个日期: {unique_dates[0]}")
                    grid_lon, grid_lat, interpolated_data, time_info = kriging_interpolation(
                        df, resolution=0.5, chunk_size=chunk_size
                    )
                    ds = save_to_netcdf(grid_lon, grid_lat, interpolated_data, output_file, time_info)
                    visualize_data(ds)
                    return ds
                    
            except Exception as e:
                print(f"日期分组处理失败: {e}")
                print("将按单一数据集处理...")
        
        # 如果没有日期列或处理失败，按单一数据集处理
        grid_lon, grid_lat, interpolated_data, time_info = kriging_interpolation(
            df, resolution=0.5, chunk_size=chunk_size
        )
        ds = save_to_netcdf(grid_lon, grid_lat, interpolated_data, output_file, time_info)
        visualize_data(ds)
        return ds
    
    except MemoryError:
        print("错误: 内存不足，无法完成处理")
        return None
    except Exception as e:
        print(f"错误: 处理失败 - {e}")
        return None

# 示例使用
if __name__ == "__main__":
    # 假设数据保存在名为'temperature_data.csv'的文件中
    input_file = "1990-2023_cy.csv"
    output_file = "temperature_interpolation.nc"
    
    # 根据系统内存情况调整chunk_size和每个区域的最大点数
    system = platform.system()
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    
    # 根据可用内存动态调整chunk_size
    if available_memory > 16000:  # 如果可用内存大于16GB
        chunk_size = 800
        max_points = 2000
    elif available_memory > 8000:  # 如果可用内存大于8GB
        chunk_size = 500
        max_points = 1500
    else:  # 内存较小
        chunk_size = 300
        max_points = 1000
    
    print(f"在{system}系统上运行，可用内存: {available_memory:.2f} MB，使用chunk_size={chunk_size}，每个区域最多处理{max_points}个点")
    
    # 运行主函数
    ds = main(input_file, output_file, chunk_size=chunk_size)

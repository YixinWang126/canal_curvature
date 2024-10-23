import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, feature
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interp1d, UnivariateSpline
import open3d as o3d  



def search_circles(data, binary_volume):
    """
    从每一层slice中的每一个连通区域找到一个或两个最大圆

    参数：
    -data: 原始数据
    -binary_volume: 与data相同形状的初始化二值掩模

    返回:
    -circles_info: 每一个元素为 {
                'slice': slice_idx,
                'region_label': region.label,
                'max_circle': {
                    'center': max_center,  # (y, x)
                    'radius': max_radius
                },
                'second_circle': {
                    'center': second_center,
                    'radius': second_radius
                },
                'third_circle': {
                    'center': third_center,
                    'radius': third_radius
                }
            }的列表
    - binary_volume: 赋值后的二值掩模
    """

    # 用于存储每个区域的最大圆和第二个圆的信息
    circles_info = []

    num_slices = data.shape[2] 

    for slice_idx in range(num_slices):
        slice_data = data[:, :, slice_idx]

        try:
            thresh = filters.threshold_otsu(slice_data)
        except:
            print(f"Slice {slice_idx}: 无法计算阈值。")
            continue
        binary = slice_data > thresh

        # 可选的形态学处理
        # binary = morphology.binary_closing(binary, morphology.disk(3))
        # binary = morphology.remove_small_objects(binary, min_size=64)

        # 将二值图像存储到三维掩模中
        binary_volume[:, :, slice_idx] = binary

        if not np.any(binary):
            print(f"Slice {slice_idx}: 未检测到根管区域。")
            continue

        # 标记连通区域
        labeled = measure.label(binary, connectivity=2)
        regions = measure.regionprops(labeled)

        if not regions:
            print(f"Slice {slice_idx}: 未找到连通区域。")
            continue

        for region in regions:
            region_mask = labeled == region.label

            # 计算距离变换
            distance = distance_transform_edt(region_mask)
            
            # 查找距离变换中的局部最大值作为候选圆心
            local_max = feature.peak_local_max(distance, footprint=np.ones((10, 10)), labels=region_mask)
            
            # 收集所有候选圆的信息（y, x, 半径）
            candidate_circles = []
            for peak in local_max:
                y, x = peak
                r = distance[y, x]
                candidate_circles.append((y, x, r))
            
            # 按半径从大到小排序候选圆
            candidate_circles = sorted(candidate_circles, key=lambda x: x[2], reverse=True)
            
            if not candidate_circles:
                print(f"Slice {slice_idx}: Region {region.label} 没有检测到圆。")
                continue
            
            # 输入最大圆
            max_circle = candidate_circles[0]
            max_center = (max_circle[0], max_circle[1])
            max_radius = max_circle[2]
            
            # 初始化第二个圆为最大圆
            second_center = max_center
            second_radius = max_radius

            # 初始化第三个圆
            third_center = max_center
            third_radius = max_radius

            first_distance = 0  # 第二圆与最大圆距离
            second_distance = 0  # 第三圆与最大圆距离
            
            # 寻找符合条件的第二个圆
            for candidate in candidate_circles[1:]:
                cand_center = (candidate[0], candidate[1])
                cand_radius = candidate[2]
                if cand_radius >= 0.3 * max_radius:
                    # 计算两个圆心之间的距离
                    dist_centers = np.sqrt((cand_center[0] - max_center[0])**2 + (cand_center[1] - max_center[1])**2)
                    if dist_centers >= (max_radius + cand_radius):
                        # 找到符合条件的第二个圆
                        second_center = cand_center
                        second_radius = cand_radius
                        first_distance = dist_centers
                        # 同步第三个圆
                        third_center = cand_center
                        third_radius = cand_radius
                        second_distance = dist_centers
                        break  

            # 找第三个圆
            for candidate in candidate_circles[1:]:
                cand_center = (candidate[0], candidate[1])
                cand_radius = candidate[2]
                if cand_radius < second_radius:  
                    if cand_radius >= 0.3 * max_radius:
                        dist_max = np.sqrt((cand_center[0] - max_center[0])**2 + (cand_center[1] - max_center[1])**2)  # 距最大圆
                        dist_second = np.sqrt((cand_center[0] - second_center[0])**2 + (cand_center[1] - second_center[1])**2)
                        if (dist_max >= (max_radius + cand_radius)) and (dist_second >= (second_radius + cand_radius)):
                            # 找到第三个圆
                            third_center = cand_center
                            third_radius = cand_radius
                            second_distance = dist_max
                            break
            
            # 替换为最远的圆
            if second_distance > first_distance:
                second_center = third_center
                second_radius = third_radius

            # 若第二个圆距离不足则排除
            if first_distance <= (9 * second_radius):
                second_center = max_center
                second_radius = max_radius

                third_center = max_center
                third_radius = max_radius
            
            # 记录该区域的圆信息
            circles_info.append({
                'slice': slice_idx,
                'region_label': region.label,
                'max_circle': {
                    'center': max_center,  # (y, x)
                    'radius': max_radius
                },
                'second_circle': {
                    'center': second_center,
                    'radius': second_radius
                },
                'third_circle': {
                    'center': third_center,
                    'radius': third_radius
                }
            })
    return circles_info, binary_volume


def find_lines(circles_info, point_max_dist=5):
    """
    从多层最大圆信息中找到连贯的中心线

    参数：
    -circles_info: 最大圆信息
    -point_max_dist: 允许一条线的相邻层最大距离

    返回：
    -central_lines: 所有找到的线
    central_lines 中形式:
        {line_1: [{'layer': j, 'point': (y, x)}, 
                {'layer': j+1, 'point': (y1, x1)}, 
                ...], 
        line_2: ...}
    """
    # 找出最大层数
    if circles_info:
        layer_list = sorted(circles_info, key=lambda x: x['slice'], reverse=True)
        max_layer = layer_list[0]['slice']
    else:
        max_layer = 0
    central_lines = {}
    '''
    central_lines 中形式:
        {line_1: [{'layer': j, 'point': (y, x)}, 
                {'layer': j+1, 'point': (y1, x1)}, 
                ...], 
        line_2: ...}
    '''

    # 遍历每一层
    for layer in range(max_layer + 1):  # 包含 max_layer
        layer_points = [p for p in circles_info if p['slice'] == layer]  # 该层中存在的中心点
        if layer_points:
            num_line = len(central_lines)
            for point in layer_points:
                point1 = point['max_circle']['center']
                point2 = point['second_circle']['center']

                # 处理最大圆中心点（point1）
                distance_dict = {}
                line1_name = ''  # 第一个点加入的中心线
                if central_lines:
                    for line_name, line in central_lines.items():
                        last_point = line[-1]['point']
                        distance = np.sqrt((last_point[0] - point1[0])**2 + (last_point[1] - point1[1])**2)
                        distance_dict[line_name] = distance

                    # 找到最小距离的线
                    if distance_dict:
                        sorted_dist = sorted(distance_dict.items(), key=lambda x: x[1])
                        shortest_line_name, shortest_dist = sorted_dist[0][0], sorted_dist[0][1]
                    else:
                        shortest_dist, shortest_line_name = float('inf'), None

                    if shortest_dist < point_max_dist and shortest_line_name is not None:
                        central_lines[shortest_line_name].append({'layer': layer, 'point': point1})
                        line1_name = shortest_line_name
                        print(f'Slice {layer} 加入最大圆中心点到 {shortest_line_name}，与上一点距离 {shortest_dist:.2f}')
                    else:
                        new_line_name = f'line_{num_line + 1}'  # 新线名
                        central_lines[new_line_name] = []
                        central_lines[new_line_name].append({'layer': layer, 'point': point1})
                        line1_name = new_line_name
                        print(f'Slice {layer} 加入最大圆中心点到新线 {new_line_name}，与其余线距离 {shortest_dist:.2f}')
                else:
                    # 创建第一条线
                    central_lines['line_1'] = []
                    central_lines['line_1'].append({'layer': layer, 'point': point1})
                    print(f'Slice {layer} 创建第一条线')
                    if point1 != point2:
                        central_lines['line_2'] = []
                        central_lines['line_2'].append({'layer': layer, 'point': point2})
                        print(f'Slice {layer} 创建第二条线')

                # 处理第二个圆中心点（point2）
                if point1 != point2:
                    distance_dict = {}
                    for line_name, line in central_lines.items():
                        last_point = line[-1]['point']
                        distance = np.sqrt((last_point[0] - point2[0])**2 + (last_point[1] - point2[1])**2)
                        distance_dict[line_name] = distance

                    if distance_dict:
                        sorted_dist = sorted(distance_dict.items(), key=lambda x: x[1])
                        shortest_line_name, shortest_dist = sorted_dist[0][0], sorted_dist[1][1] if len(sorted_dist) > 1 else (sorted_dist[0][0], sorted_dist[0][1])
                    else:
                        shortest_line_name, shortest_dist = None, float('inf')

                    if shortest_dist < point_max_dist and shortest_line_name != line1_name and shortest_line_name is not None:
                        central_lines[shortest_line_name].append({'layer': layer, 'point': point2})
                        print(f'Slice {layer} 加入次大圆中心点到 {shortest_line_name}，与上一点距离 {shortest_dist:.2f}')
                    elif shortest_dist >= point_max_dist and shortest_line_name != line1_name and shortest_line_name is not None:
                        new_line_name = f'line_{len(central_lines) + 1}'
                        central_lines[new_line_name] = []
                        central_lines[new_line_name].append({'layer': layer, 'point': point2})
                        print(f'Slice {layer} 加入次大圆中心点到新线 {new_line_name}，与其余线距离 {shortest_dist:.2f}')
    return central_lines

def interpolate_and_smooth_line(line_points, window_size=3):
    """
    对单条中心线进行插值和光滑处理。
    
    参数：
    - line_points: list of dicts, 每个 dict 包含 'layer' 和 'point'。
    - window_size: int, 平滑窗口大小。
    
    返回：
    - smooth_line: list of dicts, 插值和平滑后的中心线。
    """
    # 按 layer 排序
    sorted_points = sorted(line_points, key=lambda p: p['layer'])
    layers = [p['layer'] for p in sorted_points]
    y_coords = [p['point'][0] for p in sorted_points]
    x_coords = [p['point'][1] for p in sorted_points]
    
    # 定义完整的 layer 范围
    min_layer = min(layers)
    max_layer = max(layers)
    full_layers = np.arange(min_layer, max_layer + 1)
    
    # 创建插值函数
    interp_func_y = interp1d(layers, y_coords, kind='quadratic', fill_value='extrapolate')
    interp_func_x = interp1d(layers, x_coords, kind='quadratic', fill_value='extrapolate')
    
    # 进行插值
    interpolated_y = interp_func_y(full_layers)
    interpolated_x = interp_func_x(full_layers)

    # 使用 Savitzky-Golay 滤波器进行平滑
    # 首先确保 window_length 不超过数据长度并且为奇数
    from scipy.signal import savgol_filter
    def smooth_data(data, window_size):
        if len(data) < window_size:
            # 如果数据长度不足，调整 window_size为最大奇数不超过数据长度
            window_size = len(data) if len(data) % 2 != 0 else len(data) - 1
            if window_size < 3:
                # 数据长度太短，使用其他平滑方法，如移动平均
                window_size = max(1, len(data) // 2)
                return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        polyorder = min(3, window_size - 1)
        return savgol_filter(data, window_length=window_size, polyorder=polyorder)
    
    smooth_y = smooth_data(interpolated_y, window_size=window_size)
    smooth_x = smooth_data(interpolated_x, window_size=window_size)
    
    # 构建平滑后的中心线
    smooth_line = []
    # Adjust layers to match the smoothed data length
    adjusted_layers = full_layers[:len(smooth_y)]
    for l, y, x in zip(adjusted_layers, smooth_y, smooth_x):
        smooth_line.append({'layer': int(l), 'point': (float(y), float(x))})
    
    return smooth_line



def interpolate_and_smooth_line_b(line_points, smooth_factor=1.0):
    """
    对单条中心线进行插值和 b-Spline 平滑处理。
    
    参数：
    - line_points: list of dicts, 每个 dict 包含 'layer' 和 'point'。
    - smooth_factor: float, 平滑因子。较大的值会产生更平滑的曲线。
    
    返回：
    - smooth_line: list of dicts, 插值和平滑后的中心线。
    """
    # 按 layer 排序
    sorted_points = sorted(line_points, key=lambda p: p['layer'])
    layers = [p['layer'] for p in sorted_points]
    y_coords = [p['point'][0] for p in sorted_points]
    x_coords = [p['point'][1] for p in sorted_points]
    
    # 定义完整的 layer 范围
    min_layer = min(layers)
    max_layer = max(layers)
    full_layers = np.arange(min_layer, max_layer + 1)
    
    # 创建插值函数
    interp_func_y = interp1d(layers, y_coords, kind='linear', fill_value='extrapolate')
    interp_func_x = interp1d(layers, x_coords, kind='linear', fill_value='extrapolate')
    
    # 进行插值
    interpolated_y = interp_func_y(full_layers)
    interpolated_x = interp_func_x(full_layers)

    # 检查插值结果是否有 NaN 或 Inf
    if np.isnan(interpolated_y).any() or np.isnan(interpolated_x).any():
        print("警告: 插值结果中存在 NaN 值。")
        return []
    if np.isinf(interpolated_y).any() or np.isinf(interpolated_x).any():
        print("警告: 插值结果中存在 Inf 值。")
        return []
    
    # 使用 b-Spline 进行平滑
    def smooth_data_bSpline(layers, data, smooth_factor):
        """
        使用 b-Spline 对数据进行平滑。
        
        参数：
        - layers: array-like, 自变量。
        - data: array-like, 因变量。
        - smooth_factor: float, 平滑因子。
        
        返回：
        - smooth_data: array-like, 平滑后的数据。
        """
        # 创建 b-Spline 对象
        spline = UnivariateSpline(layers, data, s=smooth_factor)
        # 评估平滑后的数据
        smooth_data = spline(layers)
        return smooth_data
    
    # 平滑 y 和 x 坐标
    smooth_y = smooth_data_bSpline(full_layers, interpolated_y, smooth_factor)
    smooth_x = smooth_data_bSpline(full_layers, interpolated_x, smooth_factor)
    
    # 构建平滑后的中心线
    smooth_line = []
    for l, y, x in zip(full_layers, smooth_y, smooth_x):
        smooth_line.append({'layer': int(l), 'point': (float(y), float(x))})
    
    return smooth_line


def process_central_lines(central_lines, window_size=3, top_n=4):
    """
    处理所有中心线，选择最长的 top_n 条进行插值和平滑。
    
    参数：
    - central_lines: dict, 形式为 { 'line_1': [ { 'layer': j, 'point': (y, x) }, ... ], 'line_2': ... }
    - window_size: int, 平滑窗口大小。
    - top_n: int, 选择最长的 top_n 条中心线。
    
    返回：
    - central_smooth_line: dict, 形式与 central_lines 相同，包含平滑后的中心线。
    """
    # 计算每条线的长度（点数）
    sorted_lines = sorted(central_lines.items(), key=lambda x: len(x[1]), reverse=True)
    top_n_lines = sorted_lines[:top_n] 
    
    print(f"\n选择最长的 {top_n} 条中心线:")
    for i, (line_name, points) in enumerate(top_n_lines, 1):
        print(f"{i}. {line_name} - 点数: {len(points)}")
    
    # 仅对 top_n_lines 进行插值和平滑
    central_smooth_line = {}
    
    for line_name, line_points in top_n_lines:
        smooth_line = interpolate_and_smooth_line(line_points, window_size)
        central_smooth_line[line_name] = smooth_line
    
    return central_smooth_line

def process_b_spline_central_lines(central_lines, smooth_factor=1, top_n=4):
    """
    处理所有中心线，选择最长的 top_n 条进行插值和b-spline平滑。
    
    参数：
    - central_lines: dict, 形式为 { 'line_1': [ { 'layer': j, 'point': (y, x) }, ... ], 'line_2': ... }
    - window_size: int, 平滑窗口大小。
    - top_n: int, 选择最长的 top_n 条中心线。
    
    返回：
    - central_smooth_line: dict, 形式与 central_lines 相同，包含平滑后的中心线。
    """
    # 计算每条线的长度（点数）
    sorted_lines = sorted(central_lines.items(), key=lambda x: len(x[1]), reverse=True)
    top_n_lines = sorted_lines[:top_n] 
    
    print(f"\n选择最长的 {top_n} 条中心线:")
    for i, (line_name, points) in enumerate(top_n_lines, 1):
        print(f"{i}. {line_name} - 点数: {len(points)}")
    
    # 仅对 top_n_lines 进行插值和平滑
    central_smooth_line = {}
    
    for line_name, line_points in top_n_lines:
        print(f"处理{line_name}")
        smooth_line = interpolate_and_smooth_line_b(line_points, smooth_factor)
        central_smooth_line[line_name] = smooth_line
    
    return central_smooth_line


# 可视化原始和处理后的中心线以进行验证
def visualize_lines(original_lines, smooth_lines, num_samples=5):
    """
    可视化部分原始和处理后的中心线点。
    
    参数：
    - original_lines: dict, 原始中心线。
    - smooth_lines: dict, 平滑后的中心线。
    - num_samples: int, 每条线要可视化的点数。
    """
    for line_name in smooth_lines:
        original_points = original_lines[line_name][:num_samples] if line_name in original_lines else []
        smooth_points = smooth_lines[line_name][:num_samples]
        
        layers_orig = [p['layer'] for p in original_points]
        y_orig = [p['point'][0] for p in original_points]
        x_orig = [p['point'][1] for p in original_points]
        
        layers_smooth = [p['layer'] for p in smooth_points]
        y_smooth = [p['point'][0] for p in smooth_points]
        x_smooth = [p['point'][1] for p in smooth_points]
        
        plt.figure(figsize=(8, 4))
        if original_points:
            plt.scatter(x_orig, y_orig, c='r', label='Original Points')
        plt.plot(x_smooth, y_smooth, c='b', label='Smoothed Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"Visualization of {line_name}")
        plt.legend()
        plt.show()


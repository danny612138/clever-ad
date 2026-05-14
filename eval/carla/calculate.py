import json
import re
import math

def parse_coordinates(text):
    """
    从字符串中提取 (x, y) 坐标列表。
    支持格式: "(x1, y1), (x2, y2), ..." 或 "The five passing waypoint coordinates are (x1, y1)..."
    """
    pattern = r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'
    matches = re.findall(pattern, text)
    
    coords = []
    for match in matches:
        try:
            x = float(match[0])
            y = float(match[1])
            coords.append((x, y))
        except ValueError:
            continue
    return coords

def calculate_distance(p1, p2):
    """计算两点间的欧几里得距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ade_fde(pred_coords, gt_coords):
    """
    计算 ADE 和 FDE
    :param pred_coords: 预测坐标列表 [(x,y), ...]
    :param gt_coords: 真实坐标列表 [(x,y), ...]
    :return: ade, fde
    """
    if not pred_coords or not gt_coords:
        return None, None

    length = min(len(pred_coords), len(gt_coords))
    
    if length == 0:
        return None, None

    pred_slice = pred_coords[:length]
    gt_slice = gt_coords[:length]

    distances = []
    for i in range(length):
        dist = calculate_distance(pred_slice[i], gt_slice[i])
        distances.append(dist)

    ade = sum(distances) / len(distances)
    
    fde = distances[-1]

    return ade, fde

def process_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是有效的 JSON 格式")
        return

    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        print("错误: JSON 根节点必须是对象或对象数组")
        return

    total_ade = 0.0
    total_fde = 0.0
    count = 0

    print(f"{'Sample ID':<40} | {'ADE':>10} | {'FDE':>10} | {'Status'}")
    print("-" * 75)

    for item in data:
        sample_id = item.get('sample_id', 'Unknown')
        pred_text = item.get('answer_pred', '')
        gt_text = item.get('answer_gt_text', '')

        pred_coords = parse_coordinates(pred_text)
        gt_coords = parse_coordinates(gt_text)

        if len(pred_coords) < 5 or len(gt_coords) < 5:
            print(f"警告: Sample {sample_id} 解析出的点数不足5个 (Pred: {len(pred_coords)}, GT: {len(gt_coords)})")

        ade, fde = calculate_ade_fde(pred_coords, gt_coords)

        status = "OK"
        if ade is None:
            status = "Error"
            ade_val_str = "N/A"
            fde_val_str = "N/A"
        else:
            ade_val_str = f"{ade:.4f}"
            fde_val_str = f"{fde:.4f}"
            total_ade += ade
            total_fde += fde
            count += 1

        print(f"{sample_id:<40} | {ade_val_str:>10} | {fde_val_str:>10} | {status}")

    print("-" * 75)
    if count > 0:
        mean_ade = total_ade / count
        mean_fde = total_fde / count
        print(f"总体统计 (基于 {count} 个样本):")
        print(f"Mean ADE: {mean_ade:.4f}")
        print(f"Mean FDE: {mean_fde:.4f}")
    else:
        print("没有成功计算出任何有效样本的误差。")

if __name__ == "__main__":
    json_file = 'carla_results.json' 
    
    import os
    
    process_json_file(json_file)
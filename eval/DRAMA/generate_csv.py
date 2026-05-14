import os
import pandas as pd
import json
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="批量将JSON文件转换为CSV文件")
    parser.add_argument('--input_folder_path', type=str, required=True,
                        help='存放JSON文件的输入文件夹路径')
    parser.add_argument('--output_folder_path', type=str, required=True,
                        help='保存CSV文件的输出文件夹路径')

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # 创建输出文件夹（如果不存在）
    os.makedirs(args.output_folder_path, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    file_count = 0
    for file_name in os.listdir(args.input_folder_path):
        # 只处理.json后缀的文件
        if file_name.lower().endswith('.json'):  # 兼容大写后缀（如.JSON）
            json_file_path = os.path.join(args.input_folder_path, file_name)
            try:
                # 关键修复：指定UTF-8编码读取文件，解决解码错误
                with open(json_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                # 将JSON数据转换为DataFrame并保存为CSV
                df = pd.DataFrame(data)
                csv_file_name = file_name.replace('.json', '.csv').replace('.JSON', '.csv')
                csv_file_path = os.path.join(args.output_folder_path, csv_file_name)
                # 保存时指定UTF-8编码，避免CSV文件中文乱码
                df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

                file_count += 1
                print(f"✅ 成功转换: {json_file_path} -> {csv_file_path}")

            except json.JSONDecodeError as e:
                print(f"❌ JSON解析错误 {json_file_path}: {e}")
            except UnicodeDecodeError as e:
                # 备用方案：尝试用gbk编码读取（兼容少数非标准JSON文件）
                try:
                    with open(json_file_path, 'r', encoding='gbk') as file:
                        data = json.load(file)
                    df = pd.DataFrame(data)
                    csv_file_name = file_name.replace('.json', '.csv')
                    csv_file_path = os.path.join(args.output_folder_path, csv_file_name)
                    df.to_csv(csv_file_path, index=False, encoding='gbk')
                    file_count += 1
                    print(f"✅ （GBK编码）成功转换: {json_file_path} -> {csv_file_path}")
                except Exception as e2:
                    print(f"❌ 编码和解析均失败 {json_file_path}: {e2}")
            except Exception as e:
                print(f"❌ 转换失败 {json_file_path}: {e}")

    print(f"\n📊 转换完成！共处理 {file_count} 个JSON文件，结果已保存至: {args.output_folder_path}")

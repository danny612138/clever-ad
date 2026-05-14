import json
import os


def add_matched_birdview_to_carla_qa(input_json_path, output_json_path):
    """
    每张RGB图后紧跟【同文件名】的鸟瞰图
    结构示例：RGB_0045 → Birdview_0045 → RGB_0046 → Birdview_0046...
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for sample in data:
        if 'image' not in sample:
            print(f"跳过无image字段的样本：{sample.get('image_id', '未知')}")
            continue

        rgb_paths = sample['image']
        assert len(rgb_paths) == 5, "每个样本需包含5张RGB图"

        new_image_list = []
        for rgb_path in rgb_paths:
            # 关键：替换文件夹为birdview，保持文件名完全一致（如0045.jpg对应0045.jpg）
            birdview_path = rgb_path.replace('rgb_full', 'birdview')
            # 检查鸟瞰图文件是否存在（精准提示缺失的文件名）
            """if not os.path.exists(birdview_path):
                print(f"警告：样本 {sample['image_id']} 中，{rgb_path} 对应的鸟瞰图不存在 → {birdview_path}")"""

            # 按“RGB→同编号鸟瞰图”的顺序添加
            new_image_list.append(rgb_path)
            new_image_list.append(birdview_path)

        # 更新image列表
        sample['image'] = new_image_list

    # 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成！结果已保存到：{output_json_path}")


# 配置路径
INPUT_JSON = "carla_eval.json"  # 你的原始JSON
OUTPUT_JSON = "carla_qa_xg.json"  # 输出结果

if __name__ == "__main__":
    add_matched_birdview_to_carla_qa(INPUT_JSON, OUTPUT_JSON)
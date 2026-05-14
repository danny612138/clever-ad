import json
import os


def add_single_birdview_to_carla_qa(input_json_path, output_json_path):
    """
    给每个样本的5张RGB后加1张对应帧的鸟瞰图（选第5张RGB对应的鸟瞰图）
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for sample in data:
        if 'image' not in sample:
            print(f"跳过无image字段的样本：{sample.get('image_id', '未知')}")
            continue

        rgb_paths = sample['image']
        assert len(rgb_paths) == 5, "每个样本需包含5张RGB图"

        # 选第5张RGB对应的鸟瞰图（也可以选中间帧，比如第3张）
        target_rgb_path = rgb_paths[-1]  # 取最后一张RGB
        birdview_path = target_rgb_path.replace('rgb_full', 'birdview')

        # 检查Birdview文件是否存在
        if not os.path.exists(birdview_path):
            print(f"警告：Birdview文件不存在 -> {birdview_path}")

        # 新的image列表：5张RGB + 1张Birdview
        new_image_list = rgb_paths + [birdview_path]
        sample['image'] = new_image_list
        print(f"处理样本 {sample['image_id']}：原5张RGB → 现6张（5RGB+1Birdview）")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成！结果已保存到：{output_json_path}")


# 配置参数
INPUT_JSON = "carla_qa.json"
OUTPUT_JSON = "carla_qa_5rgb_1birdview.json"

if __name__ == "__main__":
    add_single_birdview_to_carla_qa(INPUT_JSON, OUTPUT_JSON)
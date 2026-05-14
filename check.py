import json
import os
from pathlib import Path


def verify_carlaqa_image_files(json_path: str, image_root_dir: str):
    """
    验证CarlaQA JSON文件中的image字段是否包含10张图片路径，且文件真实存在

    Args:
        json_path: CarlaQA数据集JSON文件路径（如Carlaqa.json）
        image_root_dir: 图片根目录（对应代码中的data_args.image_folder）
    """
    # ========== 1. 基础检查 ==========
    # 检查JSON文件是否存在
    if not os.path.exists(json_path):
        print(f"❌ 错误：JSON文件不存在 → {json_path}")
        return

    # 检查图片根目录是否存在
    image_root_dir = Path(image_root_dir)
    if not image_root_dir.exists():
        print(f"❌ 错误：图片根目录不存在 → {image_root_dir}")
        return

    # ========== 2. 读取JSON数据 ==========
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 错误：JSON文件格式无效 → {e}")
        return
    except Exception as e:
        print(f"❌ 错误：读取JSON文件失败 → {e}")
        return

    total_samples = len(data_list)
    valid_10_images = 0  # 有10张图片路径的样本数
    invalid_samples = []  # 无效样本（长度≠10）
    missing_files = []  # 存在但文件缺失的图片路径

    print(f"\n========== 开始验证 CarlaQA 数据集 ==========")
    print(f"📄 JSON文件路径：{json_path}")
    print(f"🖼️ 图片根目录：{image_root_dir}")
    print(f"📊 总样本数：{total_samples}")

    # ========== 3. 遍历验证每个样本 ==========
    for sample_idx, sample in enumerate(data_list):
        # 跳过前100个样本后停止（避免输出过多，可根据需要调整/注释）
        if sample_idx > 100:
            print(f"\n⚠️  已验证前100个样本，停止遍历（可注释此逻辑验证全部）")
            break

        # 检查是否有image字段
        if 'image' not in sample:
            print(f"\n样本[{sample_idx}]：无image字段 → 跳过")
            invalid_samples.append({
                "idx": sample_idx,
                "reason": "无image字段"
            })
            continue

        image_files = sample['image']

        # 检查image是否为列表
        if not isinstance(image_files, list):
            print(f"\n样本[{sample_idx}]：image字段不是列表 → 类型：{type(image_files)}")
            invalid_samples.append({
                "idx": sample_idx,
                "reason": f"image字段类型错误（应为list，实际为{type(image_files)}）"
            })
            continue

        # 检查列表长度是否为10
        img_count = len(image_files)
        if img_count != 10:
            print(f"\n样本[{sample_idx}]：图片数量异常 → 期望10张，实际{img_count}张")
            invalid_samples.append({
                "idx": sample_idx,
                "reason": f"图片数量={img_count}（期望10）",
                "image_files": image_files[:5]  # 只保留前5个路径，避免输出过长
            })
            continue

        # 到这里说明有10个路径，计数+1
        valid_10_images += 1

        # 验证每个图片文件是否存在
        for img_idx, img_path in enumerate(image_files):
            full_img_path = image_root_dir / img_path
            if not full_img_path.exists():
                missing_files.append({
                    "sample_idx": sample_idx,
                    "img_idx": img_idx,
                    "img_path": str(full_img_path),
                    "exists": False
                })
                print(f"  样本[{sample_idx}] 图片[{img_idx}]：❌ 文件缺失 → {full_img_path}")
            else:
                # 可选：验证文件是否为有效图片（避免空文件/损坏）
                if os.path.getsize(full_img_path) < 100:  # 小于100字节视为无效
                    missing_files.append({
                        "sample_idx": sample_idx,
                        "img_idx": img_idx,
                        "img_path": str(full_img_path),
                        "exists": True,
                        "invalid": True
                    })
                    print(f"  样本[{sample_idx}] 图片[{img_idx}]：⚠️  文件过小（可能损坏）→ {full_img_path}")

    # ========== 4. 输出验证总结 ==========
    print(f"\n========== 验证总结 ==========")
    print(f"✅ 包含10张图片路径的样本数：{valid_10_images}/{total_samples}")
    print(f"❌ 无效样本数（图片数量≠10/无image字段）：{len(invalid_samples)}")
    print(f"📛 缺失/损坏的图片文件数：{len(missing_files)}")

    # 输出无效样本详情（可选）
    if invalid_samples:
        print(f"\n⚠️  前5个无效样本详情：")
        for i, inv_sample in enumerate(invalid_samples[:5]):
            print(f"  样本[{inv_sample['idx']}]：{inv_sample['reason']}")

    # 输出缺失文件详情（可选）
    if missing_files:
        print(f"\n⚠️  前5个缺失/损坏的图片文件：")
        for i, miss_file in enumerate(missing_files[:5]):
            print(f"  样本[{miss_file['sample_idx']}] 图片[{miss_file['img_idx']}]：{miss_file['img_path']}")


# ========== 5. 运行验证（替换为你的实际路径） ==========
if __name__ == "__main__":
    # 请替换为你的实际路径！！！
    CARLAQA_JSON_PATH = "/public/home/guox/WiseAD-main/WiseAD-main/data/carla/carla_qa.json"  # 你的JSON文件路径
    #IMAGE_ROOT_DIR = "/public/home/guox/WiseAD-main/WiseAD-main/data/carla/Dataset"  # 你的图片根目录（对应data_args.image_folder）
    IMAGE_ROOT_DIR = "/public/home/guox/WiseAD-main/WiseAD-main/data/carla/DATASET"

    # 执行验证
    verify_carlaqa_image_files(
        json_path=CARLAQA_JSON_PATH,
        image_root_dir=IMAGE_ROOT_DIR
    )
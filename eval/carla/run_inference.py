import os
import json
import copy
import numpy as np
import torch
import argparse
from PIL import Image
from typing import Dict, Sequence
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
from transformers import PreTrainedTokenizer
from torch import nn
from dataclasses import dataclass
from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import disable_torch_init, tokenizer_image_token, KeywordsStoppingCriteria
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


# ========== 对齐训练集：轨迹点替换函数 ==========
def _get_waypoints(sources):
    """完全对齐训练集的轨迹点计算逻辑，替换[target_value]/[local_waypoints]为具体坐标"""
    try:
        if 'measurements' not in sources:
            raise ValueError("未找到measurements字段")

        route_frames = sources['route_frames']
        end_frame_id = sources['end_frame']
        measurements_path = sources['measurements']

        if not measurements_path or not os.path.exists(measurements_path):
            raise ValueError(f"无效的measurements路径: {measurements_path}")

        #measurements_data = json.load(open(measurements_path, 'r'))
        # 兼容数字和字符串key
        """end_frame_id_str = str(end_frame_id)
        if end_frame_id_str in measurements_data:
            meas = measurements_data[end_frame_id_str]
        elif end_frame_id in measurements_data:
            meas = measurements_data[end_frame_id]
        else:
            raise ValueError(f"end_frame_id {end_frame_id} 不在measurements中")"""
        measurements_data = json.load(open(measurements_path))
        #ego_theta = meas['theta']
        meas = measurements_data[end_frame_id]
        ego_theta = measurements_data[end_frame_id]['theta']

        # 对齐训练集：处理nan值
        if np.isnan(ego_theta):
            ego_theta = 0

        # 坐标变换矩阵（和训练集完全一致）
        local_R = np.array(
            [[np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
             [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]])
        R = np.array(
            [[np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
             [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]])

        #ego_x = meas['gps_x']
        #ego_y = meas['gps_y']
        ego_x = measurements_data[end_frame_id]['gps_x']
        ego_y = measurements_data[end_frame_id]['gps_y']
        local_future_waypoints = []

        # 生成未来5个轨迹点（和训练集逻辑一致）
        for future_frame_delta in range(1, 6):
            future_frame_id = min(end_frame_id + future_frame_delta * 5, route_frames - 1)
            """future_frame_id_str = str(future_frame_id)
            
            # 兼容key类型
            if future_frame_id_str in measurements_data:
                future_meas = measurements_data[future_frame_id_str]
            elif future_frame_id in measurements_data:
                future_meas = measurements_data[future_frame_id]
            else:
                # 训练集没有这个兜底逻辑，保持和训练集一致的严格性
                raise ValueError(f"future_frame_id {future_frame_id} 不在measurements中")"""
            future_meas = measurements_data[future_frame_id]

            future_ego_x = future_meas['gps_x']
            future_ego_y = future_meas['gps_y']

            future_waypoint = np.array([future_ego_x - ego_x, future_ego_y - ego_y])
            future_waypoint = local_R.T.dot(future_waypoint)
            future_waypoint = np.around(future_waypoint, decimals=1)
            # 反转y坐标（训练集核心逻辑）
            future_waypoint[1] = future_waypoint[1] * -1
            local_future_waypoints.append(future_waypoint)

        # 生成目标点（和训练集一致）
        target_x = meas['far_node_x']
        target_y = meas['far_node_y']
        target_waypoint = np.array([target_x - ego_x, target_y - ego_y])
        target_waypoint = R.T.dot(target_waypoint)
        target_waypoint = np.around(target_waypoint, decimals=1)
        target_waypoint[1] = target_waypoint[1] * -1

        # 替换占位符（和训练集一致）
        conversations = sources['conversations']
        if len(conversations) >= 1:
            conversations[0]['value'] = conversations[0]['value'].replace(
                '[target_value]',
                '({:.1f}, {:.1f})'.format(target_waypoint[0], target_waypoint[1])
            )
        if len(conversations) >= 2:
            conversations[1]['value'] = conversations[1]['value'].replace(
                '[local_waypoints]',
                '({:.1f}, {:.1f}), ({:.1f}, {:.1f}), ({:.1f}, {:.1f}), ({:.1f}, {:.1f}), ({:.1f}, {:.1f})'.format(
                    local_future_waypoints[0][0], local_future_waypoints[0][1],
                    local_future_waypoints[1][0], local_future_waypoints[1][1],
                    local_future_waypoints[2][0], local_future_waypoints[2][1],
                    local_future_waypoints[3][0], local_future_waypoints[3][1],
                    local_future_waypoints[4][0], local_future_waypoints[4][1],
                )
            )

        sources['conversations'] = conversations
        sources['local_future_waypoints'] = local_future_waypoints
        return sources

    except Exception as e:
        print(f"\n警告：处理轨迹点时出错: {e}")
        # 兜底逻辑（和训练集默认值对齐）
        conversations = sources.get('conversations', [])
        if len(conversations) >= 1:
            conversations[0]['value'] = conversations[0]['value'].replace('[target_value]', '(5.0,0.0)')
        if len(conversations) >= 2:
            conversations[1]['value'] = conversations[1]['value'].replace(
                '[local_waypoints]',
                '(1.0,0.0), (2.0,0.0), (3.0,0.0), (4.0,0.0), (5.0,0.0)'
            )
        sources['conversations'] = conversations
        sources['local_future_waypoints'] = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
        return sources


# ========== 对齐训练集：图片处理器 ==========
class CarlaImageProcessor:
    def __init__(self, image_aspect_ratio: str = 'pad'):
        self.image_aspect_ratio = image_aspect_ratio  # 和训练集对齐的参数

    def expand2square(self, pil_img, background_color):
        """完全对齐训练集的正方形填充逻辑"""
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def process_carla_images(self, image_files: list, image_folder: str, image_processor) -> torch.Tensor:
        """
        完全对齐训练集的图片处理逻辑：
        1. 拼接image_folder和图片路径
        2. 图片不存在时生成空白图片
        3. 裁剪图片上1/4区域
        4. 填充为正方形（根据image_aspect_ratio）
        """
        image_tensors = []
        background_color = tuple(int(x * 255) for x in image_processor.image_mean)

        assert isinstance(image_files, list), f'image files should be saved in a list'
        num_imgs = len(image_files)

        for index in range(num_imgs):
            image_file = image_files[index]
            img_path = os.path.join(image_folder, image_file)

            # 对齐训练集：图片不存在时使用空白图片
            if not os.path.exists(img_path):
                print(f"警告：图片文件不存在 {img_path}，使用空白图片")
                img = Image.new('RGB', (224, 224), color='white')
            else:
                img = Image.open(img_path).convert('RGB')

            # 对齐训练集：Carla数据集只取上1/4区域
            W, H = img.size
            img = np.array(img)
            img = img[:H // 4]
            img = Image.fromarray(img)

            # 对齐训练集：图片填充为正方形
            if self.image_aspect_ratio == 'pad':
                img = self.expand2square(img, background_color)

            # 预处理图片
            img_tensor = image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
            image_tensors.append(img_tensor)

        # 堆叠图片张量（和训练集一致）
        final_image_tensor = torch.stack(image_tensors)
        print(f"图像张量最终形状：{final_image_tensor.shape}")  # [num_imgs, 3, H, W]
        return final_image_tensor


def EvalCarla(args):
    """
    Carla数据集评估函数（完全对齐训练集预处理逻辑）
    """
    # 1. 初始化配置
    disable_torch_init()

    # 2. 安全读取JSON文件
    try:
        with open(args.eval_file, 'r', encoding='utf-8') as f:
            eval_json = json.load(f)
        print(f"✅ 成功加载JSON文件：{args.eval_file}，共{len(eval_json)}个样本")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON解析失败！行{e.lineno}，列{e.colno}，原因：{e.msg}")
        exit(1)
    except FileNotFoundError:
        print(f"\n❌ 找不到JSON文件：{args.eval_file}")
        exit(1)
    except Exception as e:
        print(f"\n❌ 加载JSON失败：{str(e)}")
        exit(1)

    eval_data = args.eval_file.split('/')[-1].split('.')[0]
    model_name = args.model_path.split('/')[-1]

    # 3. 加载模型、tokenizer、image_processor
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.load_8bit, args.load_4bit
    )
    # 初始化Carla专用图像处理器（对齐训练集参数）
    carla_img_processor = CarlaImageProcessor(image_aspect_ratio=args.image_aspect_ratio)

    # 4. 准备输出文件
    result = []
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, 'carla_results.json')
    # 清空旧文件
    with open(output_file, 'w', encoding='utf-8') as f:
        pass
    f = open(output_file, 'a', encoding='utf-8')

    # 5. 遍历评估样本
    for index, data in enumerate(eval_json):
        try:
            # 深拷贝避免修改原数据
            data_copy = copy.deepcopy(data)

            # 6. 核心步骤：替换轨迹点占位符（完全对齐训练集逻辑）
            if 'measurements' in data_copy:
                data_copy = _get_waypoints(data_copy)
                # 打印调试信息
                question_before = data['conversations'][0]['value']
                question_after = data_copy['conversations'][0]['value']
                print(f"\n📌 样本{index}占位符替换：")
                print(f"  替换前：{question_before[:80]}...")
                print(f"  替换后：{question_after[:80]}...")

            # 7. 提取基础信息
            sample_id = data_copy.get('image_id', f'sample_{index}')
            conversations = data_copy['conversations']
            question = conversations[0]['value'].strip()
            answer_gt = conversations[1]['value'].strip() if len(conversations) > 1 else ""

            # 8. 处理图像（完全对齐训练集逻辑）
            images_tensor = None
            if 'image' in data_copy:
                image_files = data_copy['image']
                images_tensor = carla_img_processor.process_carla_images(
                    image_files=image_files,
                    image_folder=args.image_folder,  # 对齐训练集的图片路径逻辑
                    image_processor=image_processor
                )
                # 适配模型输入（加batch维度 + 迁移到设备）
                images_tensor = images_tensor.unsqueeze(0).to(model.device, dtype=torch.float16)
            else:
                # 对齐训练集：无图片时生成全零张量
                crop_size = image_processor.crop_size
                images_tensor = torch.zeros(1, 1, 3, crop_size['height'], crop_size['width']).to(model.device,
                                                                                                 dtype=torch.float16)

            # 9. 构建prompt（替换<image>为模型识别的TOKEN）
            conv = conv_templates[args.conv_mode].copy()
            question = question.replace("<image>", DEFAULT_IMAGE_TOKEN)
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            # 10. 构建输入token
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            # 11. 模型推理
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=images_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            # 12. 解码结果
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f"[Warning] 样本{sample_id}：{n_diff_input_output}个token输入输出不一致")

            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip().rstrip(stop_str)

            # 13. 保存结果
            print(f"🚀 样本{sample_id} | {model_name}预测：{outputs[:100]}...\n")
            result.append({
                'sample_id': sample_id,
                'question': question,
                'answer_pred': outputs,
                'answer_gt': answer_gt,
                'image_files': data_copy.get('image', []),
                'local_future_waypoints': data_copy.get('local_future_waypoints', []),
            })

        except Exception as e:
            import traceback
            print(f"❌ 样本{index}（ID：{data.get('image_id', index)}）处理失败：{str(e)}")
            print(traceback.format_exc())
            result.append({
                'sample_id': data.get('image_id', f'sample_{index}'),
                'question': data.get('conversations', [{}])[0].get('value', ''),
                'answer_pred': 'ERROR',
                'answer_gt': '',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        # if index == 10:  # 测试时可取消注释，仅跑前10个样本
        #     break

    # 14. 保存结果
    json.dump(result, f, indent=4, ensure_ascii=False)
    f.close()
    print(f"✅ 评估完成！结果已保存至：{output_file}")


@dataclass
class EvalCarlaArguments:
    """参数类（完全对齐训练集参数）"""
    # 基础模型参数
    model_path: str = "wyddmw/WiseAD"
    conv_mode: str = "v1"
    temperature: float = 0.2
    top_p: float = None
    num_beams: int = 1
    max_new_tokens: int = 512
    load_8bit: bool = False
    load_4bit: bool = False

    # 数据相关参数（对齐训练集）
    eval_file: str = './data/carla/carla_eval.json'
    output_path: str = 'eval_results/carla/'
    image_folder: str = ''  # 训练集的图片文件夹参数
    image_aspect_ratio: str = 'pad'  # 训练集的图片填充参数


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 基础模型参数
    parser.add_argument("--model_path", type=str, default="wyddmw/WiseAD")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--load_4bit", type=bool, default=False)

    # Carla专用参数（对齐训练集）
    parser.add_argument("--eval_file", type=str, default='./data/carla/carla_eval.json')
    parser.add_argument("--output_path", type=str, default='eval_results/carla/')
    parser.add_argument("--image_folder", type=str, default='',
                        help="图片文件夹路径（和训练集保持一致）")
    parser.add_argument("--image_aspect_ratio", type=str, default='pad',
                        choices=['pad'], help="图片填充方式（仅支持pad，和训练集一致）")

    args = parser.parse_args()
    EvalCarla(args)
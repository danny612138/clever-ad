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
from transformers import PreTrainedTokenizer, GenerationConfig
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

        measurements_data = json.load(open(measurements_path, 'r', encoding='utf-8'))
        meas = measurements_data[end_frame_id]
        ego_theta = meas['theta']

        # 对齐训练集：处理nan值
        if np.isnan(ego_theta) or ego_theta is None:
            ego_theta = 0.0

        # 坐标变换矩阵（和训练集完全一致）
        local_R = np.array(
            [[np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
             [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]],
            dtype=np.float32
        )
        R = np.array(
            [[np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
             [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]],
            dtype=np.float32
        )

        ego_x = meas['gps_x']
        ego_y = meas['gps_y']
        local_future_waypoints = []

        # 生成未来5个轨迹点（和训练集逻辑一致）
        for future_frame_delta in range(1, 6):
            future_frame_id = min(end_frame_id + future_frame_delta * 5, route_frames - 1)
            future_meas = measurements_data[future_frame_id]

            future_ego_x = future_meas['gps_x']
            future_ego_y = future_meas['gps_y']

            future_waypoint = np.array([future_ego_x - ego_x, future_ego_y - ego_y], dtype=np.float32)
            future_waypoint = local_R.T.dot(future_waypoint)
            future_waypoint = np.around(future_waypoint, decimals=1)
            # 反转y坐标（训练集核心逻辑）
            future_waypoint[1] = future_waypoint[1] * -1
            local_future_waypoints.append(future_waypoint.tolist())  # 转list避免numpy类型问题

        # 生成目标点（和训练集一致）
        target_x = meas['far_node_x']
        target_y = meas['far_node_y']
        target_waypoint = np.array([target_x - ego_x, target_y - ego_y], dtype=np.float32)
        target_waypoint = R.T.dot(target_waypoint)
        target_waypoint = np.around(target_waypoint, decimals=1)
        target_waypoint[1] = target_waypoint[1] * -1

        # 替换占位符（和训练集一致）
        conversations = sources['conversations']
        if len(conversations) >= 1:
            conversations[0]['value'] = conversations[0]['value'].replace(
                '[target_value]',
                f'({target_waypoint[0]:.1f}, {target_waypoint[1]:.1f})'
            )
        if len(conversations) >= 2:
            wp_str = ", ".join([f'({wp[0]:.1f}, {wp[1]:.1f})' for wp in local_future_waypoints])
            conversations[1]['value'] = conversations[1]['value'].replace(
                '[local_waypoints]', wp_str
            )

        sources['conversations'] = conversations
        sources['local_future_waypoints'] = local_future_waypoints
        return sources

    except Exception as e:
        print(f"\n警告：处理轨迹点时出错: {str(e)}")
        # 兜底逻辑（和训练集默认值对齐）
        conversations = sources.get('conversations', [])
        if len(conversations) >= 1:
            conversations[0]['value'] = conversations[0]['value'].replace('[target_value]', '(5.0,0.0)')
        if len(conversations) >= 2:
            default_wp = "(1.0,0.0), (2.0,0.0), (3.0,0.0), (4.0,0.0), (5.0,0.0)"
            conversations[1]['value'] = conversations[1]['value'].replace('[local_waypoints]', default_wp)
        sources['conversations'] = conversations
        sources['local_future_waypoints'] = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
        return sources


class CarlaImageProcessor:
    def __init__(self,
                 image_aspect_ratio: str = 'pad',
                 device: torch.device = torch.device('cpu')):

        self.image_aspect_ratio = image_aspect_ratio
        self.device = device  # 初始化时指定设备，避免后续迁移问题

        # ========== 对齐训练集：1×1卷积层（6通道→3通道，RGB 50% + BirdView 50%） ==========
        self.channel_conv = nn.Conv2d(
            in_channels=6,
            out_channels=3,
            kernel_size=1,
            bias=False,
            device=self.device  # 直接在目标设备初始化
        )
        weight = torch.tensor([
            [0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
        ], dtype=torch.float32, device=self.device).unsqueeze(-1).unsqueeze(-1)
        self.channel_conv.weight = nn.Parameter(weight, requires_grad=False)
        # 冻结卷积层
        self.channel_conv.eval()

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

    def process_carla_data(self,
                           rgb_files: list,
                           birdview_files: list,
                           base_data_path: str,
                           image_processor) -> torch.Tensor:
        """
        仅使用RGB+鸟瞰图的处理逻辑：
        1. 每组：RGB图（裁剪上1/4） + 鸟瞰图（birdview）
        2. 拼接为6通道 → 1×1卷积降维为3通道
        3. 最终输出：[5, 3, 336, 336]
        """

        # 补全文件列表到5个（兜底逻辑）
        def pad_files(file_list, default_name):
            if not isinstance(file_list, list):
                file_list = []
            while len(file_list) < 5:
                file_list.append(default_name)
            return file_list[:5]  # 确保最多5个

        rgb_files = pad_files(rgb_files, "dummy.jpg")
        birdview_files = pad_files(birdview_files, "dummy.jpg")

        merged_tensors = []
        # 获取image_processor均值用于填充
        try:
            mean = image_processor.image_mean
        except AttributeError:
            mean = image_processor.config.image_mean
        background_color = tuple(int(x * 255) for x in mean)

        for idx in range(5):
            rgb_path = os.path.join(base_data_path, rgb_files[idx]) if not rgb_files[idx].startswith('./') else \
            rgb_files[idx]
            birdview_path = os.path.join(base_data_path, birdview_files[idx]) if not birdview_files[idx].startswith(
                './') else birdview_files[idx]

            # 1. 处理RGB图片（裁剪上1/4 + pad成正方形）
            if not os.path.exists(rgb_path) or rgb_files[idx] == "dummy.jpg":
                print(f"警告：RGB文件不存在 {rgb_path}，使用空白图")
                crop_size = image_processor.crop_size
                rgb_img = Image.new('RGB', (crop_size['width'], crop_size['height']), 'white')
            else:
                rgb_img = Image.open(rgb_path).convert('RGB')

            # RGB裁剪上1/4
            W, H = rgb_img.size
            rgb_img_np = np.array(rgb_img)[:H // 4]
            rgb_img = Image.fromarray(rgb_img_np)

            # RGB pad成正方形
            if self.image_aspect_ratio == 'pad':
                rgb_img = self.expand2square(rgb_img, background_color)

            # 预处理为张量 [3, H, W]
            rgb_tensor = image_processor.preprocess(rgb_img, return_tensors='pt')['pixel_values'][0].to(self.device)

            # 2. 处理鸟瞰图（birdview）
            if not os.path.exists(birdview_path) or birdview_files[idx] == "dummy.jpg":
                print(f"警告：鸟瞰图文件不存在 {birdview_path}，使用空白图")
                # 使用和RGB相同尺寸的空白张量
                birdview_tensor = torch.zeros_like(rgb_tensor, device=self.device)
            else:
                birdview_img = Image.open(birdview_path).convert('RGB')
                # birdview仅pad成正方形，不裁剪
                if self.image_aspect_ratio == 'pad':
                    birdview_img = self.expand2square(birdview_img, background_color)
                birdview_tensor = image_processor.preprocess(birdview_img, return_tensors='pt')['pixel_values'][0].to(
                    self.device)

            # 3. 拼接RGB+鸟瞰图为6通道
            concat_tensor = torch.cat([rgb_tensor, birdview_tensor], dim=0)

            # 4. 1×1卷积降维为3通道
            with torch.no_grad():
                merged_tensor = self.channel_conv(concat_tensor.unsqueeze(0)).squeeze(0)

            merged_tensors.append(merged_tensor)

        # 5. 堆叠5组 → [5, 3, 336, 336]
        final_tensor = torch.stack(merged_tensors)
        print(f"数据张量最终形状：{final_tensor.shape}")
        return final_tensor


def parse_image_list(image_list: list):
    """
    适配JSON新格式：解析10个元素的image列表（5帧×2，RGB+鸟瞰图）
    拆分规则：每2个为1帧，顺序[RGB路径, 鸟瞰图路径]，共5帧
    返回：rgb_files, birdview_files（各5个，与帧一一对应）
    """
    rgb_files = []
    birdview_files = []

    # 基础格式校验
    if not isinstance(image_list, list):
        print("警告：image不是列表格式，返回空列表")
        return rgb_files, birdview_files
    if len(image_list) != 10:
        print(f"警告：image列表长度为{len(image_list)}，非预期的10个（5帧×2），将尝试最大程度解析")

    # 核心拆分：每2个元素为1帧，i×2=RGB，i×2+1=鸟瞰图，共5帧
    for i in range(5):
        rgb_idx = i * 2
        bv_idx = i * 2 + 1
        # 提取RGB路径（判断索引是否有效）
        if rgb_idx < len(image_list):
            rgb_files.append(image_list[rgb_idx])
        # 提取鸟瞰图路径（判断索引是否有效）
        if bv_idx < len(image_list):
            birdview_files.append(image_list[bv_idx])

    print(f"✅ 解析image列表完成：RGB({len(rgb_files)}个)，鸟瞰图({len(birdview_files)}个)")
    return rgb_files, birdview_files


def EvalCarla(args):
    """
    Carla数据集评估函数（仅使用RGB+鸟瞰图）
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
    # 生成配置
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p if args.top_p is not None else 1.0,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 初始化Carla数据处理器
    carla_processor = CarlaImageProcessor(
        image_aspect_ratio=args.image_aspect_ratio,
        device=model.device
    )

    # 4. 准备输出文件
    result = []
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, 'carla_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        pass
    f = open(output_file, 'a', encoding='utf-8')

    # 5. 遍历评估样本
    for index, data in enumerate(eval_json):
        try:
            data_copy = copy.deepcopy(data)

            # 6. 替换轨迹点占位符
            if 'measurements' in data_copy:
                data_copy = _get_waypoints(data_copy)
                question_before = data['conversations'][0]['value'][:80] if len(
                    data.get('conversations', [])) > 0 else ""
                question_after = data_copy['conversations'][0]['value'][:80] if len(
                    data_copy.get('conversations', [])) > 0 else ""
                print(f"\n📌 样本{index}({data_copy.get('image_id', 'unknown')})占位符替换：")
                print(f"  替换前：{question_before}...")
                print(f"  替换后：{question_after}...")

            # 7. 提取基础信息
            sample_id = data_copy.get('image_id', f'sample_{index}')
            conversations = data_copy.get('conversations', [])
            question = conversations[0]['value'].strip() if len(conversations) > 0 else ""
            answer_gt = conversations[1]['value'].strip() if len(conversations) > 1 else ""

            # 8. 解析image列表并处理数据（适配10元素新格式）
            images_tensor = None
            if 'image' in data_copy:
                # 解析10元素的image列表为RGB/birdview列表（各5个）
                rgb_files, birdview_files = parse_image_list(data_copy['image'])

                # 处理RGB+鸟瞰图数据，生成[5,3,336,336]张量
                images_tensor = carla_processor.process_carla_data(
                    rgb_files=rgb_files,
                    birdview_files=birdview_files,
                    base_data_path=args.base_data_path,
                    image_processor=image_processor
                )
                # 加batch维度 [1, 5, 3, 336, 336]
                images_tensor = images_tensor.unsqueeze(0).to(model.device, dtype=torch.float16)
            else:
                # 兜底张量
                crop_size = image_processor.crop_size
                images_tensor = torch.zeros(
                    1, 5, 3, crop_size['height'], crop_size['width'],
                    device=model.device,
                    dtype=torch.float16
                )

            # 9. 构建prompt
            conv = conv_templates[args.conv_mode].copy()
            question = question.replace("<image>", DEFAULT_IMAGE_TOKEN) if question else ""
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            # 10. 构建输入token
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            # 确保二维张量
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(model.device)

            # Padding到8的倍数
            seq_len = input_ids.shape[1]
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            pad_len = (8 - (seq_len % 8)) % 8
            if pad_len > 0:
                pad_tensor = torch.full((1, pad_len), pad_token_id, device=model.device, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, pad_tensor], dim=1)

            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            # 11. 模型推理
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=images_tensor,
                    generation_config=gen_config,
                    stopping_criteria=[stopping_criteria],
                )

            # 12. 解码结果
            input_token_len = seq_len
            if len(output_ids.shape) == 1:
                output_ids = output_ids.unsqueeze(0)

            n_diff_input_output = (input_ids[:, :input_token_len] != output_ids[:, :input_token_len]).sum().item()
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
            error_info = str(e)
            traceback_str = traceback.format_exc()
            print(f"❌ 样本{index}（ID：{data.get('image_id', index)}）处理失败：{error_info}")
            print(traceback_str)
            result.append({
                'sample_id': data.get('image_id', f'sample_{index}'),
                'question': data.get('conversations', [{}])[0].get('value', ''),
                'answer_pred': 'ERROR',
                'answer_gt': '',
                'error': error_info,
                'traceback': traceback_str
            })
        # if index == 10:  # 测试时取消注释，仅跑前10个样本
        #     break

    # 14. 保存结果
    json.dump(result, f, indent=4, ensure_ascii=False)
    f.close()
    print(f"✅ 评估完成！结果已保存至：{output_file}")


@dataclass
class EvalCarlaArguments:
    """参数类（仅保留RGB+鸟瞰图）"""
    # 基础模型参数
    model_path: str = "wyddmw/WiseAD"
    conv_mode: str = "v1"
    temperature: float = 0.2
    top_p: float = None
    num_beams: int = 1
    max_new_tokens: int = 512
    load_8bit: bool = False
    load_4bit: bool = False

    # 数据相关参数
    eval_file: str = './data/carla/carla_eval.json'
    output_path: str = 'eval_results/carla/'
    base_data_path: str = './data/carla/DATASET/'  # 数据根路径（可覆盖JSON中的相对路径）
    image_aspect_ratio: str = 'pad'


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

    # 数据相关参数
    parser.add_argument("--eval_file", type=str, default='./data/carla/carla_nkt.json')
    parser.add_argument("--output_path", type=str, default='eval_results/carla_nkt')
    parser.add_argument("--base_data_path", type=str, default='./data/carla/DATASET/',
                        help="数据根路径（用于拼接相对路径）")
    parser.add_argument("--image_aspect_ratio", type=str, default='pad',
                        choices=['pad'], help="图片填充方式")

    args = parser.parse_args()
    EvalCarla(args)
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


# ========== 对齐训练集：轨迹点替换函数（兼容数字/字符串key，完全匹配训练集） ==========
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

        # 对齐训练集：处理nan值/None
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

        # 生成未来5个轨迹点（和训练集逻辑完全一致）
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
        # 兜底逻辑（和训练集默认值完全对齐）
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
    # 对齐训练集：添加雷达相关参数，和训练集DataArguments一致
    def __init__(self,
                 image_aspect_ratio: str = 'pad',
                 lidar_resolution: int = 336,
                 lidar_bev_range: float = 50.0,
                 lidar_z_min: float = -2.0,
                 lidar_z_max: float = 2.0,
                 device: torch.device = torch.device('cpu')):

        self.image_aspect_ratio = image_aspect_ratio
        self.lidar_resolution = lidar_resolution  # 雷达BEV分辨率，和RGB/BV一致
        self.lidar_bev_range = lidar_bev_range    # 雷达点云过滤范围（米）
        self.lidar_z_min = lidar_z_min            # 雷达点云最小高度
        self.lidar_z_max = lidar_z_max            # 雷达点云最大高度
        self.device = device                      # 初始化时指定设备，避免后续迁移问题

        # ========== 对齐训练集：1×1卷积层（9通道→3通道，RGB40% + BirdView30% + LiDAR30%） ==========
        self.channel_conv = nn.Conv2d(
            in_channels=9,  # 3(RGB)+3(BV)+3(LiDAR BEV) = 9通道
            out_channels=3,
            kernel_size=1,
            bias=False,
            device=self.device  # 直接在目标设备初始化，避免数据迁移
        )
        # 权重和训练集完全一致：RGB40%、鸟瞰图30%、雷达30%
        weight = torch.tensor([
            [0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0],  # 输出R通道
            [0.0, 0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0],  # 输出G通道
            [0.0, 0.0, 0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]   # 输出B通道
        ], dtype=torch.float32, device=self.device).unsqueeze(-1).unsqueeze(-1)
        self.channel_conv.weight = nn.Parameter(weight, requires_grad=False)
        # 冻结卷积层，和训练集一致
        self.channel_conv.eval()
        for param in self.channel_conv.parameters():
            param.requires_grad = False

    # ========== 对齐训练集：雷达点云转BEV伪图像（完全复用训练集代码，无任何修改） ==========
    def lidar_to_bev(self, lidar_path: str) -> np.ndarray:
        """将LiDAR点云[n×4]转换为3通道BEV伪图像[3, 336, 336]（CHW），和训练集逻辑完全一致"""
        if not os.path.exists(lidar_path):
            print(f"警告：LiDAR文件不存在 {lidar_path}，返回空白伪图像")
            return np.zeros((self.lidar_resolution, self.lidar_resolution, 3), dtype=np.float32)

        lidar_data = np.load(lidar_path)  # [n×4] x,y,z,intensity
        x = lidar_data[:, 0]
        y = lidar_data[:, 1]
        z = lidar_data[:, 2]
        intensity = lidar_data[:, 3]

        # 点云过滤，和训练集完全一致
        mask_range = (np.abs(x) <= self.lidar_bev_range) & (np.abs(y) <= self.lidar_bev_range)
        mask_z = (z >= self.lidar_z_min) & (z <= self.lidar_z_max)
        mask_intensity = (intensity >= 0) & (intensity <= 1)
        mask = mask_range & mask_z & mask_intensity
        x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

        if len(x) == 0:
            return np.zeros((self.lidar_resolution, self.lidar_resolution, 3), dtype=np.float32)

        # 坐标映射：米→像素（336×336），和训练集一致
        pix_x = ((x + self.lidar_bev_range) / (2 * self.lidar_bev_range)) * (
                self.lidar_resolution - 1)
        pix_y = ((y + self.lidar_bev_range) / (2 * self.lidar_bev_range)) * (
                self.lidar_resolution - 1)
        pix_x, pix_y = np.round(pix_x).astype(np.int32), np.round(pix_y).astype(np.int32)

        # 生成3通道BEV（密度、高度、强度），和训练集一致
        bev_density = np.zeros((self.lidar_resolution, self.lidar_resolution), dtype=np.float32)
        bev_height = np.zeros_like(bev_density)
        bev_intensity = np.zeros_like(bev_density)
        np.add.at(bev_density, (pix_y, pix_x), 1)
        np.add.at(bev_height, (pix_y, pix_x), z)
        np.add.at(bev_intensity, (pix_y, pix_x), intensity)

        # 归一化，和训练集完全一致
        bev_density = np.clip(bev_density / np.max(bev_density) if np.max(bev_density) > 0 else bev_density, 0, 1)
        bev_height = np.clip((bev_height / (bev_density + 1e-6) - self.lidar_z_min) /
                             (self.lidar_z_max - self.lidar_z_min), 0, 1)
        bev_intensity = np.clip(bev_intensity / (bev_density + 1e-6), 0, 1)

        # HWC→CHW，适配PyTorch张量格式，和训练集一致
        bev_image = np.stack([bev_density, bev_height, bev_intensity], axis=-1)
        return np.transpose(bev_image, (2, 0, 1))

    # ========== 对齐训练集：图像填充为正方形（完全复用训练集代码，无任何修改） ==========
    def expand2square(self, pil_img, background_color):
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

    # ========== 对齐训练集：三模态数据处理（RGB+BV+雷达，完全匹配训练集逻辑） ==========
    def process_carla_data(self,
                           rgb_files: list,
                           bv_files: list,
                           lidar_files: list,
                           base_data_path: str,
                           image_processor) -> torch.Tensor:
        """
        三模态数据处理逻辑（和训练集完全一致）：
        1. 逐帧处理：RGB（裁剪上1/4+pad） + BV（仅pad） + 雷达（转BEV）
        2. 单帧拼接：3+3+3 → 9通道 [9, 336, 336]
        3. 1×1卷积降维：9通道 → 3通道 [3, 336, 336]
        4. 堆叠5帧：[5, 3, 336, 336]（模型最终输入维度）
        """

        # 补全文件列表到5个（兜底逻辑，和训练集一致）
        def pad_files(file_list, default_name):
            if not isinstance(file_list, list):
                file_list = []
            while len(file_list) < 5:
                file_list.append(default_name)
            return file_list[:5]  # 确保最多5个

        rgb_files = pad_files(rgb_files, "dummy.jpg")
        bv_files = pad_files(bv_files, "dummy.jpg")
        lidar_files = pad_files(lidar_files, "dummy.npy")

        merged_image_tensors = []
        # 背景色：和图像处理器的均值一致，pad时无违和，和训练集一致
        try:
            mean = image_processor.image_mean
        except AttributeError:
            mean = image_processor.config.image_mean
        background_color = tuple(int(x * 255) for x in mean)
        crop_size = image_processor.crop_size

        # 逐帧处理三模态（5帧循环，和训练集完全一致）
        for frame_idx in range(5):
            rgb_path = rgb_files[frame_idx]
            bv_path = bv_files[frame_idx]
            lidar_path = lidar_files[frame_idx]

            # 路径拼接：兼容./开头的完整路径和纯文件名，和训练集一致
            rgb_path = os.path.join(base_data_path, rgb_path) if not rgb_path.startswith('./') else rgb_path
            bv_path = os.path.join(base_data_path, bv_path) if not bv_path.startswith('./') else bv_path
            lidar_path = os.path.join(base_data_path, lidar_path) if not lidar_path.startswith('./') else lidar_path

            # ========== 1. 处理RGB图像（和训练集一致：裁剪上1/4 + pad成正方形） ==========
            if not os.path.exists(rgb_path) or rgb_path == "dummy.jpg":
                print(f"警告：RGB文件不存在 {rgb_path}，使用空白图")
                rgb_img = Image.new('RGB', (crop_size['width'], crop_size['height']), 'white')
            else:
                rgb_img = Image.open(rgb_path).convert('RGB')
                # 核心：裁剪上1/4，和训练集完全一致
                W, H = rgb_img.size
                rgb_img_np = np.array(rgb_img)[:H // 4, :, :]
                rgb_img = Image.fromarray(rgb_img_np)
            # pad成正方形，和训练集一致
            if self.image_aspect_ratio == 'pad':
                rgb_img = self.expand2square(rgb_img, background_color)
            # 预处理为张量 [3, 336, 336]，并移到目标设备
            rgb_tensor = image_processor.preprocess(rgb_img, return_tensors='pt')['pixel_values'][0].to(self.device)

            # ========== 2. 处理鸟瞰图（和训练集一致：仅pad成正方形，无裁剪） ==========
            if not os.path.exists(bv_path) or bv_path == "dummy.jpg":
                print(f"警告：鸟瞰图文件不存在 {bv_path}，使用空白图")
                bv_tensor = torch.zeros_like(rgb_tensor, device=self.device)
            else:
                bv_img = Image.open(bv_path).convert('RGB')  # 强制转RGB，避免灰度图
                # 仅pad尺寸对齐，无其他操作，和训练集一致
                if self.image_aspect_ratio == 'pad':
                    bv_img = self.expand2square(bv_img, background_color)
                bv_tensor = image_processor.preprocess(bv_img, return_tensors='pt')['pixel_values'][0].to(self.device)

            # ========== 3. 处理雷达点云（和训练集一致：转BEV + 转张量） ==========
            if not os.path.exists(lidar_path) or lidar_path == "dummy.npy":
                print(f"警告：雷达文件不存在 {lidar_path}，使用空白BEV")
                lidar_tensor = torch.zeros_like(rgb_tensor, device=self.device)
            else:
                lidar_bev = self.lidar_to_bev(lidar_path)  # [3, 336, 336] CHW
                lidar_tensor = torch.from_numpy(lidar_bev).float().to(self.device)

            # ========== 4. 三模态通道拼接：3+3+3 → 9通道 [9, 336, 336]，和训练集一致 ==========
            concat_tensor = torch.cat([rgb_tensor, bv_tensor, lidar_tensor], dim=0)

            # ========== 5. 1×1卷积降维：9通道 → 3通道 [3, 336, 336]，和训练集一致 ==========
            with torch.no_grad():
                merged_tensor = self.channel_conv(concat_tensor.unsqueeze(0)).squeeze(0)

            # ========== 6. 加入帧列表 ==========
            merged_image_tensors.append(merged_tensor)

        # ========== 7. 堆叠5帧融合张量：[5, 3, 336, 336]（和训练集最终输出维度完全一致） ==========
        final_tensor = torch.stack(merged_image_tensors)
        print(f"三模态融合后张量最终形状：{final_tensor.shape}")  # 验证：[5,3,336,336]
        return final_tensor


# ========== 对齐训练集：解析15元素image列表（5帧×3模态，完全匹配训练集拆分规则） ==========
def parse_image_list(image_list: list):
    """
    适配训练集JSON格式：解析15个元素的image列表（5帧×3模态）
    拆分规则（和训练集CarlaQADataset完全一致）：每3个为1帧，顺序[RGB路径, 鸟瞰图路径, 雷达路径]，共5帧
    返回：rgb_files, bv_files, lidar_files（各5个，与帧一一对应）
    """
    rgb_files = []
    bv_files = []
    lidar_files = []

    # 基础格式校验（和训练集一致，要求15元素）
    if not isinstance(image_list, list):
        print("警告：image不是列表格式，返回空列表")
        return rgb_files, bv_files, lidar_files
    if len(image_list) != 15:
        print(f"严重警告：image列表长度为{len(image_list)}，非训练集要求的15个（5帧×3模态）！")

    # 核心拆分：和训练集CarlaQADataset完全一致的索引规则
    # 帧0: [0]RGB, [1]BV, [2]LiDAR | 帧1: [3]RGB, [4]BV, [5]LiDAR | ... | 帧4: [12]RGB, [13]BV, [14]LiDAR
    for i in range(5):
        rgb_idx = i * 3
        bv_idx = i * 3 + 1
        lidar_idx = i * 3 + 2
        # 提取各模态路径（判断索引是否有效，鲁棒性兜底）
        if rgb_idx < len(image_list):
            rgb_files.append(image_list[rgb_idx])
        if bv_idx < len(image_list):
            bv_files.append(image_list[bv_idx])
        if lidar_idx < len(image_list):
            lidar_files.append(image_list[lidar_idx])

    print(f"✅ 解析image列表完成：RGB({len(rgb_files)}个)，鸟瞰图({len(bv_files)}个)，雷达({len(lidar_files)}个)")
    return rgb_files, bv_files, lidar_files


def EvalCarla(args):
    """
    Carla数据集评估函数（三模态：RGB+鸟瞰图+雷达点云，和训练集完全对齐）
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
    # 生成配置（保持原有逻辑不变）
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p if args.top_p is not None else 1.0,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 初始化Carla三模态数据处理器（传入雷达参数，和训练集一致）
    carla_processor = CarlaImageProcessor(
        image_aspect_ratio=args.image_aspect_ratio,
        lidar_resolution=args.lidar_resolution,
        lidar_bev_range=args.lidar_bev_range,
        lidar_z_min=args.lidar_z_min,
        lidar_z_max=args.lidar_z_max,
        device=model.device  # 张量直接在模型设备初始化，避免CUDA错误
    )

    # 4. 准备输出文件
    result = []
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, 'carla_3modal_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        pass
    f = open(output_file, 'a', encoding='utf-8')

    # 5. 遍历评估样本
    for index, data in enumerate(eval_json):
        try:
            data_copy = copy.deepcopy(data)

            # 6. 替换轨迹点占位符（和训练集完全一致）
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

            # 8. 解析image列表并处理三模态数据（和训练集一致：15元素→3×5路径）
            images_tensor = None
            if 'image' in data_copy:
                # 解析15元素的image列表为RGB/BV/雷达列表（各5个）
                rgb_files, bv_files, lidar_files = parse_image_list(data_copy['image'])

                # 处理三模态数据，生成[5,3,336,336]融合张量
                images_tensor = carla_processor.process_carla_data(
                    rgb_files=rgb_files,
                    bv_files=bv_files,
                    lidar_files=lidar_files,
                    base_data_path=args.base_data_path,
                    image_processor=image_processor
                )
                # 加batch维度 [1, 5, 3, 336, 336]，适配模型输入
                images_tensor = images_tensor.unsqueeze(0).to(model.device, dtype=torch.float16)
            else:
                # 无有效三模态数据时的兜底逻辑（和训练集一致）
                crop_size = image_processor.crop_size
                images_tensor = torch.zeros(
                    1, 5, 3, crop_size['height'], crop_size['width'],
                    device=model.device,
                    dtype=torch.float16
                )

            # 9. 构建prompt（保持原有逻辑不变）
            conv = conv_templates[args.conv_mode].copy()
            question = question.replace("<image>", DEFAULT_IMAGE_TOKEN) if question else ""
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            # 10. 构建输入token（保持原有逻辑不变）
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            # 确保二维张量
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(model.device)

            # Padding到8的倍数（保持原有逻辑不变）
            seq_len = input_ids.shape[1]
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            pad_len = (8 - (seq_len % 8)) % 8
            if pad_len > 0:
                pad_tensor = torch.full((1, pad_len), pad_token_id, device=model.device, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, pad_tensor], dim=1)

            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            # 11. 模型推理（保持原有逻辑不变，输入三模态融合张量）
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=images_tensor,
                    generation_config=gen_config,
                    stopping_criteria=[stopping_criteria],
                )

            # 12. 解码结果（保持原有逻辑不变）
            input_token_len = seq_len
            if len(output_ids.shape) == 1:
                output_ids = output_ids.unsqueeze(0)

            n_diff_input_output = (input_ids[:, :input_token_len] != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f"[Warning] 样本{sample_id}：{n_diff_input_output}个token输入输出不一致")

            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip().rstrip(stop_str)

            # 13. 保存三模态评估结果
            print(f"🚀 样本{sample_id} | {model_name}三模态预测：{outputs[:100]}...\n")
            result.append({
                'sample_id': sample_id,
                'question': question,
                'answer_pred': outputs,
                'answer_gt': answer_gt,
                'image_files': data_copy.get('image', []),
                'local_future_waypoints': data_copy.get('local_future_waypoints', []),
                'model_name': model_name,
                'modal': 'RGB+BirdView+LiDAR'  # 标记三模态
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
                'traceback': traceback_str,
                'modal': 'RGB+BirdView+LiDAR'
            })
        # if index == 10:  # 测试时取消注释，仅跑前10个样本
        #     break

    # 14. 保存三模态评估结果
    json.dump(result, f, indent=4, ensure_ascii=False)
    f.close()
    print(f"✅ 三模态评估完成！结果已保存至：{output_file}")


@dataclass
class EvalCarlaArguments:
    """参数类（三模态：添加雷达相关参数，和训练集DataArguments完全一致）"""
    # 基础模型参数
    model_path: str = "wyddmw/WiseAD"
    conv_mode: str = "v1"
    temperature: float = 0.2
    top_p: float = None
    num_beams: int = 1
    max_new_tokens: int = 512
    load_8bit: bool = False
    load_4bit: bool = False

    # 数据相关参数（和训练集一致）
    eval_file: str = './data/carla/carla_eval.json'
    output_path: str = 'eval_results/carla_3modal/'
    base_data_path: str = './data/carla/DATASET/'  # 数据根路径（可覆盖JSON中的相对路径）
    image_aspect_ratio: str = 'pad'

    # 雷达相关参数（和训练集CarlaQADataset完全一致，不可随意修改）
    lidar_resolution: int = 336    # 雷达BEV分辨率，和RGB/BV一致
    lidar_bev_range: float = 50.0  # 雷达点云过滤范围（米）
    lidar_z_min: float = -2.0      # 雷达点云最小高度
    lidar_z_max: float = 2.0       # 雷达点云最大高度


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
    parser.add_argument("--eval_file", type=str, default='./data/carla/carla_ld.json')
    parser.add_argument("--output_path", type=str, default='eval_results/carla_ld')
    parser.add_argument("--base_data_path", type=str, default='./data/carla/DATASET/',
                        help="数据根路径（用于拼接相对路径）")
    parser.add_argument("--image_aspect_ratio", type=str, default='pad',
                        choices=['pad'], help="图片填充方式，和训练集一致仅支持pad")

    # ========== 添加强达相关命令行参数（和训练集完全一致） ==========
    parser.add_argument("--lidar_resolution", type=int, default=336,
                        help="雷达BEV伪图像分辨率，和训练集一致为336")
    parser.add_argument("--lidar_bev_range", type=float, default=50.0,
                        help="雷达点云过滤范围（米），和训练集一致为50.0")
    parser.add_argument("--lidar_z_min", type=float, default=-2.0,
                        help="雷达点云最小高度（米），和训练集一致为-2.0")
    parser.add_argument("--lidar_z_max", type=float, default=2.0,
                        help="雷达点云最大高度（米），和训练集一致为2.0")

    args = parser.parse_args()
    EvalCarla(args)
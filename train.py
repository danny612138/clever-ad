import os
import copy
import json
import logging
import pathlib
import numpy as np
import transformers
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from torch import nn

# ========== 关键修改0：最顶部禁用wandb环境变量 ==========
os.environ["WANDB_DISABLED"] = "true"

from mobilevlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN
from mobilevlm.train.trainer import VLMTrainer
from mobilevlm import conversation as conversation_lib
from mobilevlm.model.mobilellama import MobileLlamaForCausalLM
from mobilevlm.utils import tokenizer_image_token

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ========== 新增：自定义Loss计算函数 ==========
def compute_waypoint_loss(pred_waypoints, target_waypoints, valid_waypoints):
    """
    计算轨迹点预测损失
    pred_waypoints: [batch_size, 5, 2]
    target_waypoints: [batch_size, 5, 2]
    valid_waypoints: [batch_size, 5, 2]
    """
    # 只计算有效轨迹点的损失
    mask = valid_waypoints.bool()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_waypoints.device)

    # MSE损失
    mse_loss = F.mse_loss(pred_waypoints[mask], target_waypoints[mask], reduction='mean')
    return mse_loss


# ========== 新增：修改VLMTrainer以支持轨迹点损失 ==========
class CustomVLMTrainer(VLMTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写损失计算逻辑，同时计算LLM损失和轨迹点损失
        """
        # 提取轨迹点标签（避免被model forward处理）
        local_future_waypoints = inputs.pop("local_future_waypoints", None)
        valid_future_waypoints = inputs.pop("valid_future_waypoints", None)

        # 计算原始LLM损失
        outputs = model(**inputs)
        llm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=model.device)

        # 计算轨迹点损失（如果有）
        waypoint_loss = torch.tensor(0.0, device=llm_loss.device)
        if (local_future_waypoints is not None and valid_future_waypoints is not None and
                hasattr(model, 'agent_head') and model.agent_head is not None):
            # 获取模型最后一层输出用于预测轨迹点
            last_hidden_state = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.logits
            pred_waypoints = model.agent_head(last_hidden_state)
            waypoint_loss = compute_waypoint_loss(
                pred_waypoints, local_future_waypoints, valid_future_waypoints
            )

        # 组合损失（保留LLM loss权重，避免loss为0）
        if hasattr(model, 'llm_loss_weight') and hasattr(model, 'agent_loss_weight'):
            total_loss = model.llm_loss_weight * llm_loss + model.agent_loss_weight * waypoint_loss
        else:
            # 没有agent head时只使用LLM损失
            total_loss = llm_loss if waypoint_loss.item() == 0 else 0.1 * llm_loss + 0.9 * waypoint_loss

        # 打印损失分解（仅rank0，每100步打印一次）
        if self.state.global_step % 100 == 0 and local_rank == 0:
            print(f"\nStep {self.state.global_step}:")
            print(f"  LLM Loss: {llm_loss.item():.6f}")
            print(f"  Waypoint Loss: {waypoint_loss.item():.6f}")
            print(f"  Total Loss: {total_loss.item():.6f}")

        return (total_loss, outputs) if return_outputs else total_loss


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_tower_type: Optional[str] = field(default='clip')
    task_name: str = field(default='llm')
    train_agent: bool = field(default=True)  # 默认开启agent训练
    no_loading_pretrained_llm: bool = field(default=False)
    # 新增：轨迹点预测head配置
    waypoint_pred_dim: int = field(default=10)  # 5个点 × 2维坐标


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = field(default=True)  # 默认开启多模态
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'pad'  # 默认使用pad模式处理图片
    image_grid_pinpoints: Optional[str] = field(default=None)
    dataset_name: str = 'Carla'  # 默认使用Carla数据集
    lingoqa_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    drama_data_repeat: int = 1
    carla_repeat_factor: int = 2
    drama_data_path: str = field(default=None)
    # 新增：调试参数
    debug_print_samples: bool = field(default=True, metadata={"help": "打印样本调试信息"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)  # 必须保留False，否则会丢失轨迹点数据
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=1024,  # 增大最大长度，避免截断标签
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = field(default=False)  # 默认关闭LoRA，简化调试
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    report_to: List[str] = field(default_factory=list,
                                 metadata={"help": "Disable all experiment trackers including wandb"})
    # 新增：训练参数调整
    per_device_train_batch_size: int = field(default=4)
    learning_rate: float = field(default=5e-5)
    num_train_epochs: float = field(default=3.0)
    logging_steps: int = field(default=10)  # 更频繁地打印loss
    gradient_checkpointing: bool = field(default=False)  # 默认关闭，简化调试


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.state_dict().items(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    """【关键修复】彻底修复mask逻辑，确保GPT回答完全不被mask"""
    cur_idx = 0
    # 遍历所有token长度和说话人
    for i, (tokenized_len, speaker) in enumerate(zip(tokenized_lens, speakers)):
        if speaker == "human" or speaker == "system" or i == 0:  # 只mask人类/系统输入和header
            target[cur_idx:cur_idx + tokenized_len] = IGNORE_INDEX
        # GPT的回答不mask
        cur_idx += tokenized_len
    # 截断后的部分也mask
    if cur_idx < len(target):
        target[cur_idx:] = IGNORE_INDEX


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations],
                            dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)

    # 【关键修复】只mask图片token部分，保留回答部分
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
        # 确保回答部分不被mask
        target[tokenized_len:] = target[tokenized_len:]  # 显式保留

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        debug: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)

    conversations = []
    all_speakers = []
    all_tokenized_lens = []

    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

        # 记录每个部分的token长度和说话人
        if has_image:
            tokenized_lens = [len(tokenizer_image_token(header, tokenizer))]
            tokenized_lens += [len(tokenizer_image_token(s["value"], tokenizer)) for s in source]
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]

        speakers = ["system"] + [s["from"] for s in source]
        all_tokenized_lens.append(tokenized_lens)
        all_speakers.append(speakers)

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)

    # 应用mask逻辑
    for idx, (target, source, tokenized_lens, speakers) in enumerate(
            zip(targets, sources, all_tokenized_lens, all_speakers)):
        _mask_targets(target, tokenized_lens, speakers)

        # 调试信息：打印每个样本的mask情况
        if debug and idx < 2:  # 只打印前2个样本
            non_ignore = (target != IGNORE_INDEX).sum().item()
            total = len(target)
            print(f"\n调试样本 {idx}:")
            print(f"  总token数: {total}, 有效标签数: {non_ignore}")
            print(f"  有效标签比例: {non_ignore / total:.2%}")
            if non_ignore == 0:
                print(f"  警告：该样本无有效标签！")
                print(f"  原始对话: {source}")

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        image = None
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
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

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # 预处理数据，启用调试
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            debug=self.data_args.debug_print_samples)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        # 调试：检查labels
        if self.data_args.debug_print_samples and i < 5:
            non_ignore_count = (data_dict["labels"] != IGNORE_INDEX).sum().item()
            print(f"\nLazySupervisedDataset 样本 {i}:")
            print(f"  有效标签数: {non_ignore_count}")
            print(f"  Input IDs长度: {len(data_dict['input_ids'])}")
            print(f"  Labels长度: {len(data_dict['labels'])}")

        return data_dict


class CarlaQADataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(CarlaQADataset, self).__init__()
        list_data_dict = json.load(open(data_path, 'r'))
        rank0_print(f"加载Carla数据集，共{len(list_data_dict)}个样本，重复{data_args.carla_repeat_factor}次")
        self.tokenizer = tokenizer
        if 'carla' in data_path.lower():
            self.list_data_dict = list_data_dict * data_args.carla_repeat_factor
        else:
            self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.debug_printed = False  # 避免重复打印

        # ========== 新增：定义1×1卷积层（6通道→3通道） ==========
        self.channel_conv = torch.nn.Conv2d(
            in_channels=6,
            out_channels=3,
            kernel_size=1,
            bias=False
        )
        # 初始化权重：RGB和BEV通道平均融合（可根据需求调整）
        self.channel_conv.weight.data = torch.tensor([
            [0.5, 0.0, 0.0, 0.5, 0.0, 0.0],  # 输出R = 0.5*RGB_R + 0.5*BEV_R
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.0],  # 输出G = 0.5*RGB_G + 0.5*BEV_G
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]  # 输出B = 0.5*RGB_B + 0.5*BEV_B
        ]).float().unsqueeze(-1).unsqueeze(-1)
        # 冻结卷积层（仅作为特征融合，不参与训练）
        self.channel_conv.eval()
        for param in self.channel_conv.parameters():
            param.requires_grad = False

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _get_waypoints(self, sources):
        route_frames = sources['route_frames']
        end_frame_id = sources['end_frame']
        measurements = sources['measurements']
        measurements = json.load(open(measurements))
        ego_theta = measurements[end_frame_id]['theta']
        conversations = sources['conversations']
        processed_data = {}
        local_R = np.array(
            [[np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
             [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]])

        if np.isnan(ego_theta):
            ego_theta = 0
        R = np.array(
            [[np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
             [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]])
        ego_x = measurements[end_frame_id]['gps_x']
        ego_y = measurements[end_frame_id]['gps_y']
        local_future_waypoints = []

        # generate the local future waypoints
        for future_frame_delta in range(1, 6):
            future_frame_id = min(end_frame_id + future_frame_delta * 5, route_frames - 1)
            future_ego_x = measurements[future_frame_id]['gps_x']
            future_ego_y = measurements[future_frame_id]['gps_y']
            future_waypoint = np.array([future_ego_x - ego_x, future_ego_y - ego_y])
            future_waypoint = local_R.T.dot(future_waypoint)
            future_waypoint = np.around(future_waypoint, decimals=1)
            # inverse the y coordinates
            future_waypoint[1] = future_waypoint[1] * -1
            local_future_waypoints.append(future_waypoint)

        # generate the target waypoint
        target_x = measurements[end_frame_id]['far_node_x']
        target_y = measurements[end_frame_id]['far_node_y']
        target_waypoint = np.array([target_x - ego_x, target_y - ego_y])
        target_waypoint = R.T.dot(target_waypoint)
        target_waypoint = np.around(target_waypoint, decimals=1)
        target_waypoint[1] = target_waypoint[1] * -1

        # update prompts
        conversations[0]['value'] = conversations[0]['value'].replace('[target_value]',
                                                                      '({:.1f}, {:.1f})'.format(target_waypoint[0],
                                                                                                target_waypoint[1]))
        conversations[1]['value'] = conversations[1]['value'].replace(
            '[local_waypoints]',
            '({:.1f}, {:.1f}), ({:.1f}, {:.1f}), ({:.1f}, {:.1f}), ({:.1f}, {:.1f}), ({:.1f}, {:.1f})'
            .format(local_future_waypoints[0][0], local_future_waypoints[0][1],
                    local_future_waypoints[1][0], local_future_waypoints[1][1],
                    local_future_waypoints[2][0], local_future_waypoints[2][1],
                    local_future_waypoints[3][0], local_future_waypoints[3][1],
                    local_future_waypoints[4][0], local_future_waypoints[4][1],
                    ))
        sources['conversations'] = conversations
        print(conversations[1]['value'])

        # update numerical waypoints for agent learning
        sources['local_future_waypoints'] = local_future_waypoints

    def __rmul__(self, v):
        self.list_data_dict = v * self.list_data_dict
        return self

    # ========== 新增：复用的正方形填充函数 ==========
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

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i].copy()  # 避免修改原数据

        if 'measurements' in sources.keys():
            self._get_waypoints(sources)

        if self.data_args.debug_print_samples and not self.debug_printed:
            print(f"\n=== CarlaQADataset 第一个样本详情 ===")
            print(f"原始数据: {sources}")
            print(f"对话内容: {sources.get('conversations', [])}")
            self.debug_printed = True

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        # 处理图片
        image = None
        if 'image' in sources[0]:
            image_files = sources[0]['image']
            assert isinstance(image_files, list), f'image files should be saved in a list'
            assert len(image_files) == 10, f"样本{i}需要10张图片（5RGB+5BEV），实际{len(image_files)}张"

            num_groups = 5  # 5组（RGB+BEV）
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            merged_image_tensors = []  # 存储每组融合后的3通道张量
            background_color = tuple(int(x * 255) for x in processor.image_mean)

            # ========== 核心修改：按RGB+BEV成对处理10张图片 ==========
            for group_idx in range(num_groups):
                # 1. 获取当前组的RGB和BEV图片路径
                rgb_idx = group_idx * 2
                bev_idx = group_idx * 2 + 1
                rgb_file = image_files[rgb_idx]
                bev_file = image_files[bev_idx]

                # 2. 加载并处理RGB图片（裁剪上1/4，和原有逻辑一致）
                rgb_path = os.path.join(image_folder, rgb_file)
                if not os.path.exists(rgb_path):
                    print(f"警告：RGB图片不存在 {rgb_path}，使用空白图片")
                    rgb_img = Image.new('RGB', (336, 336), color='white')
                else:
                    rgb_img = Image.open(rgb_path).convert('RGB')

                # RGB图片裁剪上1/4
                W, H = rgb_img.size
                rgb_img_np = np.array(rgb_img)
                rgb_img_np = rgb_img_np[:H // 4]  # 只取上1/4
                rgb_img = Image.fromarray(rgb_img_np)

                # RGB图片填充为正方形
                if self.data_args.image_aspect_ratio == 'pad':
                    rgb_img = self.expand2square(rgb_img, background_color)

                # RGB图片预处理为张量 [3, 336, 336]
                rgb_tensor = processor.preprocess(rgb_img, return_tensors='pt')['pixel_values'][0]

                # 3. 加载并处理BEV图片（不裁剪，保持完整）
                bev_path = os.path.join(image_folder, bev_file)
                if not os.path.exists(bev_path):
                    print(f"警告：BEV图片不存在 {bev_path}，使用空白图片")
                    bev_img = Image.new('RGB', (336, 336), color='white')
                else:
                    bev_img = Image.open(bev_path).convert('RGB')

                # BEV图片填充为正方形（不裁剪）
                if self.data_args.image_aspect_ratio == 'pad':
                    bev_img = self.expand2square(bev_img, background_color)

                # BEV图片预处理为张量 [3, 336, 336]
                bev_tensor = processor.preprocess(bev_img, return_tensors='pt')['pixel_values'][0]

                # 4. 拼接RGB+BEV为6通道张量 [6, 336, 336]
                concat_tensor = torch.cat([rgb_tensor, bev_tensor], dim=0)

                # 5. 1×1卷积降维为3通道 [3, 336, 336]
                with torch.no_grad():  # 卷积层不训练，禁用梯度
                    merged_tensor = self.channel_conv(concat_tensor.unsqueeze(0)).squeeze(0)

                # 6. 添加到融合张量列表
                merged_image_tensors.append(merged_tensor)

            # 7. 堆叠5组融合后的张量，得到最终形状 [5, 3, 336, 336]
            image = torch.stack(merged_image_tensors)

            # 轨迹点数据处理（原有逻辑不变）
            local_future_waypoints = np.array(sources[0].get('local_future_waypoints', [[0.0, 0.0]] * 5))
            local_future_waypoints = torch.from_numpy(local_future_waypoints).float()
            valid_future_waypoints = torch.ones((5, 2)).float()

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            # 无图片时的兜底逻辑（原有逻辑不变）
            sources = copy.deepcopy([e["conversations"] for e in sources])
            local_future_waypoints = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]).float()
            valid_future_waypoints = torch.ones((5, 2)).float()

        # 原有预处理逻辑（不变）
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in sources[0]),
            debug=self.data_args.debug_print_samples)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

            # 调试：检查labels是否全为IGNORE_INDEX
            non_ignore_count = (data_dict["labels"] != IGNORE_INDEX).sum().item()
            if non_ignore_count == 0:
                print(f"\n警告：样本{i}的labels全部为IGNORE_INDEX！")
                print(f"原始对话：{sources[0].get('conversations', [])}")
                # 强制设置一些有效标签
                if len(data_dict["labels"]) > 10:
                    data_dict["labels"][-10:] = data_dict["input_ids"][-10:]  # 最后10个token作为有效标签
            elif self.data_args.debug_print_samples and i < 3:
                print(f"\n样本{i}的有效标签数：{non_ignore_count}")
                # 解码前20个token查看
                input_text = self.tokenizer.decode(data_dict["input_ids"][:20], skip_special_tokens=True)
                label_text = self.tokenizer.decode(data_dict["labels"][data_dict["labels"] != IGNORE_INDEX][:20],
                                                   skip_special_tokens=True)
                print(f"  输入前20token: {input_text}")
                print(f"  有效标签: {label_text}")

        # 添加图片数据
        if 'image' in sources[0]:
            data_dict['image'] = image  # 最终形状 [5, 3, 336, 336]
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(5, 3, crop_size['height'], crop_size['width'])  # 适配新形状

        # 添加轨迹点数据（确保这些数据不会被TrainingArguments移除）
        data_dict['local_future_waypoints'] = local_future_waypoints
        data_dict['valid_future_waypoints'] = valid_future_waypoints

        return data_dict


class LingoQADataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LingoQADataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        image = None
        if 'image' in sources[0]:
            image_files = sources[0]['image']
            assert isinstance(image_files, list), f'image files should be saved in a list'
            num_imgs = len(image_files)
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image_tensors = []

            for index in range(num_imgs):
                image_file = image_files[index]
                img_path = os.path.join(image_folder, image_file)
                if not os.path.exists(img_path):
                    print(f"警告：图片文件不存在 {img_path}，使用空白图片")
                    img = Image.new('RGB', (224, 224), color='white')
                else:
                    img = Image.open(img_path).convert('RGB')

                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
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

                    img = expand2square(img, tuple(int(x * 255) for x in processor.image_mean))
                    img_tensor = processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
                else:
                    img_tensor = processor.preprocess(img, return_tensors='pt')['pixel_values'][0]

                image_tensors.append(img_tensor)

            image = torch.stack(image_tensors)
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in sources[0]),
            debug=self.data_args.debug_print_samples)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        if image is not None:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        # 调试信息
        if self.data_args.debug_print_samples and i < 2:
            non_ignore_count = (data_dict["labels"] != IGNORE_INDEX).sum().item()
            print(f"\nLingoQADataset 样本 {i}:")
            print(f"  有效标签数: {non_ignore_count}")

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """【修复】修复轨迹点批量处理逻辑"""
    tokenizer: transformers.PreTrainedTokenizer
    debug_printed: bool = False  # 调试标记

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        # 对input_ids和labels进行padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)

        # 截断到最大长度
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        # 构建基础batch
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 处理图片数据
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # 【修复】正确堆叠轨迹点数据（原代码只取第一个样本）
        if 'local_future_waypoints' in instances[0]:
            batch["local_future_waypoints"] = torch.stack([
                instance['local_future_waypoints'] for instance in instances
            ])
            batch['valid_future_waypoints'] = torch.stack([
                instance['valid_future_waypoints'] for instance in instances
            ])

        # 调试：打印第一个batch的信息
        if not self.debug_printed:
            print(f"\n=== 第一个Batch信息 ===")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Labels shape: {batch['labels'].shape}")
            print(f"有效标签总数: {(batch['labels'] != IGNORE_INDEX).sum().item()}")
            self.debug_printed = True

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, ) -> Dict:
    """构建数据模块"""
    if data_args.dataset_name == 'LingoQA':
        train_dataset = LingoQADataset(tokenizer=tokenizer,
                                       data_path=data_args.data_path,
                                       data_args=data_args)
    elif data_args.dataset_name == 'Carla':
        train_dataset = CarlaQADataset(tokenizer=tokenizer,
                                       data_path=data_args.data_path,
                                       data_args=data_args)
    elif data_args.dataset_name == 'joint_LingoQA_Carla':
        carla_data_path = data_args.data_path
        carla_dataset = CarlaQADataset(tokenizer=tokenizer,
                                       data_path=carla_data_path,
                                       data_args=data_args)
        lingoqa_data_path = data_args.lingoqa_data_path
        lingoqa_dataset = LingoQADataset(tokenizer=tokenizer,
                                         data_path=lingoqa_data_path,
                                         data_args=data_args)
        train_dataset = ConcatDataset([carla_dataset, lingoqa_dataset])
    elif data_args.dataset_name == 'joint_LingoQA_Carla_DRAMA':
        carla_data_path = data_args.data_path
        carla_dataset = CarlaQADataset(tokenizer=tokenizer,
                                       data_path=carla_data_path,
                                       data_args=data_args)
        lingoqa_data_path = data_args.lingoqa_data_path
        lingoqa_dataset = LingoQADataset(tokenizer=tokenizer,
                                         data_path=lingoqa_data_path,
                                         data_args=data_args)
        drama_data_path = data_args.drama_data_path
        drama_dataset = CarlaQADataset(tokenizer=tokenizer,
                                       data_path=drama_data_path,
                                       data_args=data_args)
        drama_dataset = data_args.drama_data_repeat * drama_dataset
        train_dataset = ConcatDataset([carla_dataset, lingoqa_dataset, drama_dataset])
    else:
        raise ValueError(f"不支持的数据集名称：{data_args.dataset_name}")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


# ========== 新增：轨迹点预测Head ==========
class WaypointPredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim] 或 [batch_size, vocab_size]
        if len(x.shape) == 3:
            x = x[:, -1, :]  # 使用最后一个token的输出预测轨迹点
        elif len(x.shape) == 2:
            # 如果是logits，取最后一维的均值（备用方案）
            x = x.mean(dim=1)
        # 前向传播
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x.view(-1, 5, 2)  # [batch_size, 5, 2]


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 强制禁用wandb
    training_args.report_to = []
    # 关键：确保不移除自定义列（轨迹点数据）
    training_args.remove_unused_columns = False
    local_rank = training_args.local_rank

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 量化配置
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))

    # 加载模型
    model = None
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            raise ValueError("MPT模型暂不支持多模态")
        else:
            if training_args.local_rank == 0:
                print(f"加载模型：{model_args.model_name_or_path}")
            if not model_args.no_loading_pretrained_llm:
                model = MobileLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            else:
                print('不加载预训练LLM权重')
                model_copy = MobileLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
                model = MobileLlamaForCausalLM(model_copy.config)
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir, **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False

    # ========== 关键：添加轨迹点预测Head ==========
    if model_args.train_agent:
        # 获取隐藏层维度
        hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
        # 添加agent head
        model.agent_head = WaypointPredictionHead(hidden_dim, model_args.waypoint_pred_dim)
        # 设置loss权重（确保LLM loss不为0）
        model.llm_loss_weight = 0.1
        model.agent_loss_weight = 0.9
        print(f"添加轨迹点预测Head，隐藏维度：{hidden_dim}")
        print(f"Loss权重 - LLM: {model.llm_loss_weight}, Agent: {model.agent_loss_weight}")

        # 将agent head移到正确设备
        if training_args.device != 'cpu':
            model.agent_head = model.agent_head.to(training_args.device)

    # 冻结backbone（如果需要）
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
        # 但确保agent head可训练
        if hasattr(model, 'agent_head'):
            model.agent_head.requires_grad_(True)

    # 4/8bit量化训练准备
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # 梯度检查点
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # LoRA配置
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("添加LoRA适配器...")
        model = get_peft_model(model, lora_config)

    # 加载tokenizer
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    # 设置pad token
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 多模态配置
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.vision_tower_type = training_args.vision_tower_type = model_args.vision_tower_type

        if not training_args.lora_enable:
            model.requires_grad_(True)

        # 只微调mm_projector
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            # 同时确保agent head可训练
            if hasattr(model, 'agent_head'):
                model.agent_head.requires_grad_(True)

        # 冻结mm_projector
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        # 量化模式下调整mm_projector dtype
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        # 图片token配置
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # 调整量化模型的dtype
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = VLMTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            print('non_lora_trainable...', non_lora_state_dict.keys())
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
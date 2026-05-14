import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 1. 配置路径 ==========
BASE_MODEL_PATH = "/public/home/liuwx/WiseAD-main/WiseAD-main/train/time8lora/epoch1/mobilevlm_v2-2.finetune"
LORA_ADAPTER_PATH = "/public/home/liuwx/WiseAD-main/WiseAD-main/train/time8lora/epoch1/mobilevlm_v2-2.finetune-lora"
MERGED_MODEL_SAVE_PATH = "/public/home/liuwx/WiseAD-main/WiseAD-main/train/time8lora/epoch1/mobilevlm_v2-2-merged"  # 合并后保存的路径

# ========== 2. 加载基座模型 + LoRA适配器 ==========
# 加载基座模型（不量化，确保合并后权重完整）
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    #device_map="cpu",  # 合并时用CPU，避免GPU显存不足
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 挂载LoRA
lora_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

# ========== 3. 合并权重并卸载LoRA ==========
merged_model = lora_model.merge_and_unload()

# ========== 4. 保存合并后的完整模型 ==========
merged_model.save_pretrained(
    MERGED_MODEL_SAVE_PATH,
    safe_serialization=True,  # 推荐用safetensors格式，更安全
    max_shard_size="10GB"  # 大模型分片保存，避免单文件过大
)

# 保存tokenizer（必须，否则推理时找不到）
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH)

print(f"✅ 合并完成！完整模型已保存至：{MERGED_MODEL_SAVE_PATH}")
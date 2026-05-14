import sys
import argparse
from PIL import Image
from pathlib import Path
import os
import json
import base64
from io import BytesIO
# 千问API所需库
from openai import OpenAI

# 新增：图片转Base64编码（千问VL API要求，支持单图/多图）
def image_to_base64(image_path):
    """
    将本地图片文件转为千问VL API要求的Base64编码字符串（带MIME前缀）
    :param image_path: 本地图片绝对/相对路径
    :return: 带MIME前缀的base64编码字符串（如data:image/jpeg;base64,xxx）
    """
    try:
        with Image.open(image_path).convert('RGB') as img:
            buffer = BytesIO()
            # 保存为JPG格式（兼容性好，减小编码体积）
            img.save(buffer, format='JPEG', quality=90)
            # 转Base64并解码为字符串 + 添加MIME前缀（核心修复点）
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            # 关键：添加千问要求的MIME类型前缀
            base64_with_prefix = f"data:image/jpeg;base64,{base64_str}"
        return base64_with_prefix
    except Exception as e:
        raise Exception(f"图片转Base64失败：{image_path} | 错误：{str(e)}")

def EvalLingoQA(args):
    # 1. 读取LingoQA评估数据（完全保留原代码逻辑）
    try:
        eval_json = json.load(open(args.eval_file, 'r', encoding='utf-8'))
        print(f"✅ 成功加载评估文件：{args.eval_file}，共{len(eval_json)}个样本")
    except Exception as e:
        print(f"❌ 加载评估文件失败：{str(e)}")
        return
    eval_data = args.eval_file.split('/')[-1].split('.')[0]
    model_name = args.model  # 千问模型名（如qwen2.5-vl-7b-instruct）
    result = []

    # 2. 准备结果保存文件（完全保留原代码逻辑）
    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, 'lingoqa_qwen_results.json')
    else:
        save_path = 'lingoqa_qwen_results.json'
    # 清空旧文件（避免追加重复内容）
    with open(save_path, 'w', encoding='utf-8') as f:
        pass
    f = open(save_path, 'a', encoding='utf-8')

    # 3. 初始化千问API客户端（兼容环境变量/手动指定API Key）
    try:
        client = OpenAI(
            api_key=args.api_key if args.api_key else os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # 验证API Key是否有效（简单校验）
        if not client.api_key or not client.api_key.startswith('sk-'):
            raise Exception("API Key无效！请检查是否以sk-开头")
        print(f"✅ 千问API客户端初始化成功，使用模型：{model_name}")
    except Exception as e:
        print(f"❌ 千问API客户端初始化失败：{str(e)}")
        print("💡 解决方案：1. 配置环境变量DASHSCOPE_API_KEY 2. 手动指定--api_key sk-xxx")
        f.close()
        return

    # 4. 遍历LingoQA样本（保留原代码遍历/字段提取逻辑，替换推理部分）
    for index, data in enumerate(eval_json):
        # 初始化单样本结果（兜底值）
        sample_res = {
            'question_id': data.get('question_id', f'q_{index}'),
            'question': data['conversations'][0]['value'].strip(),
            'segment_id': data.get('image_id', f'seg_{index}'),
            'answer': 'API_CALL_FAILED',
            'answer_gt': data['conversations'][1]['value'].strip(),
            'error': None
        }
        try:
            # 提取原代码核心字段（完全保留）
            segment_id = data['image_id']
            question_id = data['question_id']
            question = data['conversations'][0]['value'].strip()
            answer_gt = data['conversations'][1]['value'].strip()
            print(f"\n📌 样本{index} | QID:{question_id} | SEG:{segment_id}")
            print(f"问题：{question[:50]}...")

            # 5. 处理图片（兼容原代码list/str格式，转千问要求的Base64）
            image_base64_list = []
            if isinstance(data['image'], list):
                # 多图：按原代码排序后依次转Base64
                image_lists = sorted(data['image'])
                for img_path in image_lists:
                    b64 = image_to_base64(img_path)
                    image_base64_list.append(b64)
                print(f"处理多图：{len(image_base64_list)}张")
            else:
                # 单图：直接转Base64
                b64 = image_to_base64(data['image'])
                image_base64_list.append(b64)
                print(f"处理单图：{data['image']}")

            # 6. 构建千问VL API的messages（核心适配，支持单图/多图）
            user_content = []
            # 先加所有图片（千问要求：image对象在前，文本在后）
            for b64_str in image_base64_list:
                user_content.append({
                    "type": "image_url",  # 核心修复点：千问VL API要求用image_url而非image
                    "image_url": {
                        "url": b64_str  # 传入带MIME前缀的Base64字符串
                    }
                })
            # 再加文本问题
            user_content.append({
                "type": "text",
                "text": question
            })
            # 千问标准messages格式
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant. Answer the question based on the given images.'},
                {'role': 'user', 'content': user_content}
            ]

            # 7. 调用千问VL API推理（适配原代码超参：temperature/max_new_tokens等）
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p if args.top_p else 1.0,
                max_tokens=args.max_new_tokens,  # 对应原代码max_new_tokens
                n=1
            )
            # 提取API回答
            answer_pred = completion.choices[0].message.content.strip()
            print(f"🚀 千问{model_name}回答：{answer_pred[:100]}...")

            # 8. 更新单样本结果（成功时）
            sample_res = {
                'question_id': question_id,
                'question': question,
                'segment_id': segment_id,
                'answer': answer_pred,
                'answer_gt': answer_gt,
                'error': None
            }

        except Exception as e:
            # 单样本异常兜底：记录错误信息，不中断整体评估
            error_info = str(e)
            sample_res['error'] = error_info
            print(f"❌ 样本{index}处理失败：{error_info[:100]}...")

        # 收集结果（无论成功/失败都加入，保证样本数完整）
        result.append(sample_res)

    # 9. 保存最终结果（完全保留原代码的JSON格式，缩进4格）
    json_data = json.dumps(result, indent=4, ensure_ascii=False)
    f.write(json_data)
    f.close()
    print(f"\n✅ 评估完成！结果保存至：{save_path}")
    # 统计成功/失败样本数
    success_num = len([r for r in result if r['error'] is None])
    fail_num = len(result) - success_num
    print(f"📊 统计：总样本{len(result)} | 成功{success_num} | 失败{fail_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ===================== 千问API相关参数（新增/修改）=====================
    parser.add_argument("--model", type=str, default="qwen2.5-vl-7b-instruct",
                        help="千问VL模型名，可选：qwen2.5-vl-7b-instruct/qwen2.5-vl-14b-instruct")
    parser.add_argument("--api_key", type=str, default='sk-8e13f36749aa4348bc0637e4e6468ebc',
                        help="阿里云DashScope API Key（sk-开头），优先使用环境变量DASHSCOPE_API_KEY")
    # ===================== 保留原代码的推理超参（适配API）=====================
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="采样温度，0为贪心解码")
    parser.add_argument("--top_p", type=float, default=None,
                        help="核采样，None则使用1.0")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大生成token数，对应API的max_tokens")
    # ===================== 保留原代码的LingoQA数据参数（完全不变）=====================
    parser.add_argument("--eval_file", type=str, default='data/evaluation_data.json',
                        help="LingoQA评估JSON文件路径")
    parser.add_argument("--output_path", type=str, default='result/',
                        help="结果保存路径")
    # ===================== 移除本地模型无关参数（load_8bit/4bit/conv_mode）=====================

    args = parser.parse_args()
    EvalLingoQA(args)
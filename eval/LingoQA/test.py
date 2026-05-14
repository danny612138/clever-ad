from CIDEr import CIDEr  # 只导入CIDEr类即可，无需手动导入tokenize

# ==================== 修正后的测试数据 ====================
# 核心：
# - predictions：预测文本列表（单层列表，每个元素是一个预测字符串）
# - references：参考文本列表（双层列表，每个元素是一个样本的参考文本列表）
predictions = ["a cat on the mat"]  # 预测文本（对应你的res）
references = [["a cat is on the mat"]]  # 参考文本（对应你的gts，必须是双层列表）

# ==================== 初始化CIDEr实例 ====================
c_score = CIDEr()
cider_score = c_score.compute(predictions=predictions, references=references)

# ==================== 输出结果 ====================
print("✅ 评估结果：", cider_score)
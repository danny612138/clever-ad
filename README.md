<h1 align="center">🚗 Clever-AD: 多输入模态增强的自动驾驶系统</h1>

<p align="center">
基于知识增强端到端自动驾驶框架，扩展支持<b>鸟瞰图输入</b>、<b>雷达输入</b>等多种模态，提供开箱即用的不同输入版本。
</p>

## 📦 分支说明（核心！）
本仓库采用Git分支管理不同输入版本，切换对应分支即可使用对应功能：

| 分支名称 | 功能说明 | 适用场景 |
|---------|---------|---------|
| main | clever-AD基础版本 | 标准视觉输入的问答与轨迹预测 |
| feature/birdview | 新增鸟瞰图输入支持 | 结合鸟瞰图的全局路径规划 |
| feature/radar | 新增雷达输入支持 | 恶劣天气下的障碍物检测与避障 |

## 🦙 模型与数据
- 基础模型：MobileVLM V2 1.7B
- 微调数据集：[LingoQA](https://github.com/wayveai/LingoQA)、[DRAMA](https://usa.honda-ri.com/drama)、[Carla](https://github.com/opendilab/LMDrive)多模态数据集


## 🛠️ 安装步骤
### 1. 克隆仓库
根据需要选择克隆对应分支：
```bash
# 克隆基础版本（默认main分支）
git clone https://github.com/danny612138/clever-ad.git
cd clever-ad

# 直接克隆鸟瞰图版本
git clone -b feature/birdview https://github.com/danny612138/clever-ad.git
cd clever-ad
```
### 2. 创建并激活 conda 环境
```bash
# 安装环境
conda create -n cleverad python=3.10 -y
conda activate cleverad
pip install -r requirements.txt
```

## 🪜 Training & Evaluation
数据集准备
* [CarlaDataset](https://huggingface.co/datasets/OpenDILabCommunity/LMDrive)
* [DRAMA](https://usa.honda-ri.com/drama)
* [LingoQA](https://github.com/wayveai/LingoQA)

同时，参考Wise-AD的[json文件](https://huggingface.co/datasets/wyddmw/WiseAD_training_data)，数据集结构如下：
```bash
data
├── carla
│   ├── DATASET
│   │   ├── routes_town01_long_w1...
│   │   └── routes_town01_long_w2...
│   └── carla_qa.json
├── DRAMA
│   ├── drama_data
│   │    ├── combined
│   │    │   ├── 2020-0127-132751
│   │    │   ├── 2020-0129-105040
│   │    │   └── ...
│   └── DRAMA_qa.json
├── LingoQA
│   ├── action
│   │   └── images
│   ├── evaluation
│   │   └── images
│   ├── scenery
│   │   └── images
│   ├── training_data.json
│   └── evaluation_data.json
```
# 启动训练
```bash
bash launch.sh
```
# 在 LingoQA 数据集上进行评估：
```bash
sh eval/LingoQA/eval_lingoqa.sh /path/to/WiseAD/checkpoint /path/to/save/predictions
# An example: 
# sh eval/LingoQA/eval_lingoqa.sh wyddmw/WiseAD /home/spyder/WiseAD/eval_results
```
# 🙏 致谢
* 感谢原作者开源的[WiseAD](https://github.com/wyddmw/WiseAD)项目
* 感谢 [MobileVLM](https://github.com/Meituan-AutoML/MobileVLM) 和 [LMDrive](https://github.com/opendilab/LMDrive) 项目提供的基础框架

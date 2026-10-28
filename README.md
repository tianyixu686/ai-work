# 基于产生式系统与轻量 CNN 的动物识别

作者：徐天一

本作业对比两条技术路线：

1. 产生式系统：将颜色、边缘、纹理和形状统计量离散成事实，用 IF–THEN 规则正向链推理。
2. 轻量 CNN：使用深度可分离卷积与残差连接，端到端学习四类动物特征。

目标类别为猫、狗、鹿、马，对应 CIFAR-10 中的 `cat/dog/deer/horse`。代码与参考作业的结构、类别范围、规则定义和 CNN 网络均不同。

## 目录

- `动物识别+徐天一.docx`：完整报告。
- `code/expert_system.py`：产生式推理机和图像事实提取。
- `code/model.py`：轻量深度可分离残差 CNN。
- `code/data.py`：数据下载、四类筛选与数据增强。
- `code/train.py`：训练、验证、早停和模型保存。
- `code/predict.py`：单图规则推理与 CNN 预测。
- `examples/`：四张 CIFAR-10 测试样例，用于演示。

## 快速运行

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 规则系统可独立运行，不需要训练 CNN
python code/expert_system.py examples/cat.png

# 训练 CNN（首次运行自动下载 CIFAR-10）
python code/train.py --epochs 20 --batch-size 128

# 联合预测
python code/predict.py examples/dog.png --weights results/best_model.pt
```

## 实验说明

脚本将原始训练集按固定随机种子分为训练集和验证集，原始测试集只用于最终评估。训练结束后会在 `results/`中保存最优权重、训练日志和测试指标。报告不伪造未运行的数值，其中的表格给出了应记录的指标及分析方法。

## 数据来源

CIFAR-10 官方页面：<https://www.cs.toronto.edu/~kriz/cifar.html>。数据仅用于课程学习。


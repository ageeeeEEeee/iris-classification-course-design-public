# 基于深度学习的鸢尾花识别分类

《人工智能编程基础》综合设计。使用 TensorFlow 多层感知机，根据以厘米记录的萼片和花瓣长宽识别三种鸢尾花。

## 运行

实测环境为 Windows x64、Python 3.13.14、TensorFlow 2.20.0 和 scikit-learn 1.7.2，使用 CPU。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe predict.py 5.1 3.5 1.4 0.2
.\.venv\Scripts\python.exe train.py
```

四个位置参数依次为萼片长度、萼片宽度、花瓣长度和花瓣宽度，单位均为厘米。预测脚本可直接使用仓库中的 `results/iris.keras`。示例返回 `setosa` 及三类概率。重新训练会覆盖 `results/` 下的实验文件。

## 实验结果

scikit-learn 的 Iris 数据集共 150 条记录。固定种子 42 分两次分层抽样，得到训练集 90 条、验证集 30 条、测试集 30 条。标准化仅在训练集拟合。共训练 92 轮，保留第 72 轮的权重。

| 模型 | 正确条数 | 测试准确率 | 宏平均 F1 | 测试对数损失 |
| --- | ---: | ---: | ---: | ---: |
| 多层感知机 | 28/30 | 93.33% | 0.9333 | 0.1258 |
| 逻辑回归 | 28/30 | 93.33% | 0.9333 | 0.2067 |

两条神经网络错分记录发生在 versicolor 和 virginica 之间。这里的指标描述本次划分，无法代表在其他采样条件下的表现。

## 成果文件

- `综合设计报告-鸢尾花识别分类.docx` 为课程报告。
- `requirements.txt` 列出直接依赖，`requirements-lock.txt` 记录实测环境的依赖版本。
- `train.py` 和 `predict.py` 分别用于训练评估和单样本推理。
- `results/metrics.json` 记录划分索引、学习曲线数据、逐类指标及错分记录。
- `results/evaluation.png` 汇总学习曲线和测试集混淆矩阵。
- `results/iris.keras` 包含已训练网络和归一化参数。

数据出处见 [UCI Iris](https://doi.org/10.24432/C56C76) 和 [scikit-learn 数据集说明](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)。实验通过 scikit-learn 的 `load_iris()` 读取修订版数据。

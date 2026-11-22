# 鸢尾花照片识别

选择一张 JPG、PNG、WebP 或 BMP 照片。程序先给出「可能是鸢尾花」「不像鸢尾花」或「无法确定」，再对可能的鸢尾花尝试识别 *Iris setosa*、*Iris versicolor* 和 *Iris virginica*。当最高分对应其他鸢尾，或目标种得分低于阈值时，品种栏显示「未确定」。低分目标种可作为待复核候选。

图片在本机读取和识别。程序不会上传图片，实际预测无需联网。

## 启动

在项目目录运行以下命令安装环境。Python 3.13 使用 Windows 官方发行版时，请保留 Tkinter 组件。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

安装后双击 `Start-Photo-Recognition.cmd`，在窗口中点击「选择图片」。先用 `photo/examples/iris.jpg` 和 `photo/examples/not-iris.jpg` 检查两种判断，再选择自己的照片。命令行也可运行 `.venv\Scripts\python.exe photo/identify.py photo/examples/iris.jpg`。

## 照片实验

从 iNaturalist 的研究级观察中收集 399 张采用 CC0 或 CC BY 许可的照片。按观察者分组划分训练、验证、测试集合，分别为 249、71、79 张。使用预训练的 EfficientNetV2B0 提取图像特征，训练是否为鸢尾属的二分类器和包含「其他鸢尾」的四分类器。

界面阈值下，79 张测试照片中有 63 张的鸢尾判断正确，7 张显示「无法确定」。界面共给出 15 个明确品种名，其中 9 个与标签一致。品种结果应结合照片主体、花部细节和其他植物特征复核。分数是模型输出，未做概率校准。由于测试规模有限，结果不能代表所有种类、拍摄角度或背景。

完整统计见 [metrics.json](metrics.json)，照片来源、作者、许可和划分见 [image_sources.csv](image_sources.csv)。复现实验时先运行 `python photo/collect.py` 按清单下载照片和权重，再运行 `python photo/train_photo.py`。训练脚本会更新模型文件和统计结果。

## 演示照片署名

- [`iris.jpg`](examples/iris.jpg) 为 *Iris setosa* 的照片。(c) Pavel Smirnov, some rights reserved (CC BY)。[原观察](https://www.inaturalist.org/observations/236746822)，[许可](https://creativecommons.org/licenses/by/4.0/)。演示副本转换为 JPEG 并缩放。
- [`not-iris.jpg`](examples/not-iris.jpg) 为 *Hemerocallis fulva* 的照片。(c) Tom Field, some rights reserved (CC BY)。[原观察](https://www.inaturalist.org/observations/296661903)，[许可](https://creativecommons.org/licenses/by/4.0/)。演示副本转换为 JPEG 并缩放。

照片采集依据 [iNaturalist API](https://api.inaturalist.org/v1/docs/)，特征网络来自 [Keras EfficientNetV2](https://keras.io/api/applications/efficientnet_v2/)。

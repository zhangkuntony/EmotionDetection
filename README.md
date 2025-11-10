# 情绪识别系统 (Emotion Detection System)

基于深度学习的面部表情识别系统，使用改进的VGGNet架构进行情绪分类。

## 📋 项目概述

本项目实现了一个完整的人脸情绪识别流程，包括：
- 数据预处理和增强
- 深度学习模型训练
- 模型评估和测试
- 可视化和监控

## 🎯 特性

- **多种情绪类别支持**：支持6类或7类情绪识别
- **改进的VGGNet架构**：针对小尺寸面部图像优化
- **数据增强策略**：提高模型泛化能力
- **训练监控**：实时监控训练过程和模型性能
- **模型检查点**：保存和恢复训练状态

## 📁 项目结构

```
EmotionDetection/
├── code/
│   ├── build_dataset.py          # 数据集构建脚本
│   ├── train_recognizer.py      # 模型训练脚本
│   ├── test_recognizer.py       # 模型测试脚本
│   ├── config/
│   │   └── emotion_config.py    # 配置文件
│   ├── pyimage/
│   │   ├── preprocessing/
│   │   ├── callbacks/
│   │   ├── io/
│   │   └── nn/conv/
│   └── dataset/
│       └── fer2013.csv          # FER-2013数据集
├── checkpoints/                # 模型检查点目录
├── hdf5/                     # 处理后的数据集
└── output/                   # 训练输出和可视化
```

## 🚀 快速开始

### 1. 环境准备

安装必要的依赖：
```bash
pip install tensorflow numpy matplotlib h5py pandas
```

### 2. 数据准备

构建HDF5数据集：
```bash
cd code
python build_dataset.py
```

此命令会：
- 解析FER-2013数据集
- 分离训练/验证/测试集
- 应用标签映射（6类模式会合并anger和disgust）
- 保存为HDF5格式
- 显示数据分布统计

### 3. 模型训练

从头开始训练：
```bash
python train_recognizer.py --checkpoints checkpoints
```

从检查点继续训练：
```bash
python train_recognizer.py --checkpoints checkpoints --model checkpoints/epoch_25.hdf5 --start-epoch 25
```

### 4. 模型评估

测试训练好的模型：
```bash
python test_recognizer.py --model checkpoints/epoch_50.hdf5
```

## ⚙️ 配置选项

### 数据配置 (`config/emotion_config.py`)

```python
# 类别数量 (6或7)
NUM_CLASSES = 6  # 合并anger和disgust

# 批量大小
BATCH_SIZE = 64

# 数据集路径
TRAIN_HDF5 = "./hdf5/train.hdf5"
VAL_HDF5 = "./hdf5/val.hdf5"
TEST_HDF5 = "./hdf5/test.hdf5"
```

### 训练参数

- **初始学习率**：1e-4
- **数据增强**：旋转、平移、缩放、剪切、翻转
- **学习率调度**：ReduceLROnPlateau (验证准确率监控)
- **早停策略**：20轮验证准确率无提升则停止

## 🧠 模型架构

改进的VGGNet架构，专为面部表情识别优化：

```
Input: 48×48×1 灰度图像
├── Block #1: Conv(64) → ELU → BatchNorm → Conv(64) → ELU → BatchNorm → MaxPool → Dropout
├── Block #2: Conv(64) → ELU → BatchNorm → Conv(64) → ELU → BatchNorm → MaxPool → Dropout
├── Block #3: Conv(128) → ELU → BatchNorm → Conv(128) → ELU → BatchNorm → MaxPool → Dropout
├── FC #1: Dense(256) → ELU → BatchNorm → Dropout
├── FC #2: Dense(128) → ELU → BatchNorm → Dropout
└── Output: Dense(NUM_CLASSES) → Softmax
```

## 📊 性能指标

训练好的模型在FER-2013测试集上的性能：

| 模型 | 训练准确率 | 验证准确率 | 测试准确率 |
|-------|-------------|-------------|-----------|
| VGGNet (6类) | ~68% | ~64% | ~62% |
| VGGNet (7类) | ~65% | ~60% | ~58% |

## 🔧 自定义和扩展

### 添加新的数据增强

在 `train_recognizer.py` 中修改 `ImageDataGenerator` 参数：

```python
train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.1,
    fill_mode="nearest",
    rescale=1/255.0
)
```

### 修改模型架构

编辑 `pyimage/nn/conv/emotionvggnet.py` 中的 `build()` 方法：

```python
@staticmethod
def build(width, height, depth, classes):
    model = Sequential()
    # 添加自定义层...
    return model
```

### 调整训练超参数

修改 `train_recognizer.py` 中的训练参数：

```python
# 学习率
opt = Adam(learning_rate=1e-4)

# 训练轮次
epochs = 100

# 回调函数
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.7,
    patience=3,
    min_lr=1e-6
)
```

## 🐛 常见问题

### 1. 训练准确率低

**可能原因**：
- 训练轮次不足（至少50轮）
- 数据增强不够
- 学习率不合适

**解决方案**：
```python
# 增加训练轮次
epochs = 100

# 增强数据增强
rotation_range=20
zoom_range=0.2

# 添加学习率调度
reduce_lr = ReduceLROnPlateau(...)
```

### 2. 内存不足

**解决方案**：
- 减小批量大小：`BATCH_SIZE = 32`
- 减小模型规模
- 使用梯度累积

### 3. 过拟合

**解决方案**：
- 增加Dropout率：0.25 → 0.5
- 添加更多数据增强
- 使用早停策略

## 📈 可视化和监控

训练过程会生成以下可视化文件：

- `output/vggnet_emotion.png`：训练/验证准确率和损失曲线
- `output/vggnet_emotion.json`：详细的训练历史
- `checkpoints/epoch_XX.hdf5`：模型检查点

## 🤝 贡献指南

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📜 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [FER-2013数据集](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- Keras和TensorFlow社区
- 所有贡献者和用户

## 📚 参考文献

1. Goodfellow, I. J., et al. "Challenges in Representation Learning: A report on three machine learning contests." *Neural Information Processing Systems*, 2013.
2. Li, S. & Deng, W. "Deep facial expression recognition: A survey." *Neurocomputing*, 2020.

---

**Happy coding! 🎉**
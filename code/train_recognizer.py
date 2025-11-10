# USAGE
# python train_recognizer.py --checkpoints checkpoints
# python train_recognizer.py --checkpoints checkpoints --model checkpoints/epoch_40.hdf5 --start-epoch 10

# set the matplotlib backend so figures can be saved in the background
# 工具包
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import emotion_config as config
from pyimage.preprocessing import ImageToArrayPreprocessor
from pyimage.callbacks import EpochCheckpoint
from pyimage.callbacks import TrainingMonitor
from pyimage.io import HDF5DatasetGenerator
from pyimage.nn.conv.emotionvggnet import EmotionVGGNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as backend
import argparse
import os

# construct the argument parse and parse the arguments
# 命令行参数
ap = argparse.ArgumentParser()
# checkpoint: 在网络训练过程中将权重进行保存
ap.add_argument("-c", "--checkpoints", required=True, help="path to checkpoints directory")
# 指定获取哪个具体的checkpoint
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
# 指定当前开始训练的epoch
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the training and testing image generators for data
# augmentation, then initialize the image preprocessor
# 图像增强
train_aug = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")
val_aug = ImageDataGenerator(rescale=1 / 255.0)
# 实例化
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
# 得到训练和验证的数据的生成器
train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=train_aug, preprocessors=[iap], classes=config.NUM_CLASSES)
val_gen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=val_aug, preprocessors=[iap], classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
# 未指定具体的checkpoint，从头训练
if args["model"] is None:
    print("[INFO] compiling model ...")
    model = EmotionVGGNet.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
    opt = Adam(learning_rate=1e-4)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 指定了具体的checkpoint
else:
    print("[INFO] loading {}...".format(args["model"]))
    # 加载checkpoint
    model = load_model(args["model"])

    # 获取当前参数
    print("[INFO] old learning rate: {}".format(backend.get_value(model.optimizer.learning_rate)))
    # 设置新的学习率（更安全的方法：重新编译模型）
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-5), metrics=["accuracy"])
    print("[INFO] new learning rate: 1e-5 (模型已重新编译)")

# 定义回调函数
figPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.json"])
callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5, start_at=args["start_epoch"]),
    TrainingMonitor(figPath, jsonPath, start_at=args["start_epoch"])
]

# 训练网络
model.fit(
    train_gen.generator(),
    steps_per_epoch=train_gen.numImages // config.BATCH_SIZE,
    validation_data=val_gen.generator(),
    validation_steps=val_gen.numImages // config.BATCH_SIZE,
    epochs=75,
    callbacks=callbacks,
    verbose=1
)

# 关闭数据流
train_gen.close()
val_gen.close()
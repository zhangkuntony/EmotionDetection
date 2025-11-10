# USAGE
# python test_recognizer.py --model checkpoints/epoch_25.hdf5

# import the necessary packages
from config import emotion_config as config
from pyimage.preprocessing import ImageToArrayPreprocessor
from pyimage.io import HDF5DatasetGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import argparse

# 加载模型：命令行参数指定
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="path to model checkpoint to load")
args = vars(ap.parse_args())

# initialize the testing data generator and image preprocessor
# 图像的像素进行归一化
test_aug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

# initialize the testing dataset generator
# 获取测试数据
test_gen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE, aug=test_aug, preprocessors=[iap], classes=config.NUM_CLASSES)

# 加载模型
print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

# 预测
(loss, acc) = model.evaluate(
    test_gen.generator(),
    steps=test_gen.numImages // config.BATCH_SIZE,
    verbose=1
)
print("[INFO] accuracy: {:.2f}%".format(acc * 100))

# close the testing database
# 读取数据流关闭
test_gen.close()
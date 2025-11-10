

from config import emotion_config as config
from pyimage.io import HDF5DatasetWriter

import numpy as np

# open the input file for reading (skipping the header), then
# initialize the list of data and labels for the training,
# validation, and testing sets
print("[INFO] loading input data...")
f = open(file=config.INPUT_PATH)
f.__next__()

# 创建空的列表
(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

# loop over the rows in the input file
# 遍历所有行
for row in f:
    # extract the label, image, and use from the row
    # 获取每一行的数据
    (label, image, usage) = row.strip().split(",")
    # label
    label = int(label)

    # if wre are ignoring the "disgust" class there will be 6 total classes labels instead of 7
    # 因为我们忽略了disgust这一类，所以将第二类1和第一类0进行合并
    if config.NUM_CLASSES == 6:
        # merge together the "anger" and "disgust" classes
        if label == 1:
            label = 0

        # if label has a value greater than zero, subtract one from it
        # to make all labels sequential (not required, but helps when
        # interpreting results)
        # 其他类别进行相应的调整
        if label > 0:
            label -= 1

    # reshape the flattened pixel list into a 48*48 (grayscale)
    # 将扁平话的像素列表重置为48*48的灰度图像
    image = np.array(image.split(" "), dtype="uint8")
    image = image.reshape((48, 48))

    # 根据类型，写入相应的hdf5文件中
    # check if we are examining a training image
    if usage == "Training":
        trainImages.append(image)
        trainLabels.append(label)

    # check if this is a validation image
    elif usage == "PrivateTest":
        valImages.append(image)
        valLabels.append(label)

    # otherwise, this must be a testing image
    else:
        testImages.append(image)
        testLabels.append(label)

# construct a list pairing the training, validation, and testing
# images along with their corresponding labels and output HDF5 files
# 构建list, 数据+hdf5的路径
datasets = [
    (trainImages, trainLabels, config.TRAIN_HDF5),
    (valImages, valLabels, config.VAL_HDF5),
    (testImages, testLabels, config.TEST_HDF5)
]

# loop over the dataset tuples
# 遍历数据集 dataset 元组
for (images, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    # 实例化写入的类
    writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)

    # loop over the image and add them to the dataset
    # 遍历图像和label，写入
    for (image, label) in zip(images, labels):
        writer.add([image], [label])

    # close the HDF5 writer
    # 关闭
    writer.close()

# close the input file
f.close()
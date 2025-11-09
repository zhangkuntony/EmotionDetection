# import the necessary packages
import h5py
import os

class HDF5DatasetWriter:
    # 初始化：dims指定存储数据的shape[N, H, W], hdf文件的位置
    def __init__(self, dims, output_path, data_key="images", buf_size=1000):
        # check to see if the output path exists,
        # and if so, raise an exception
        # 判断输出路径是否存在，若存在，则抛出异常
        if os.path.exists(output_path):
            raise ValueError("The supplied `output_path` already "
				"exists and cannot be overwritten. Manually delete "
				"the file before continuing.", output_path)

        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class labels
        # 打开HDF5文件
        self.db = h5py.File(output_path, "w")
        # 创建存储图像和label的空间
        self.data = self.db.create_dataset(data_key, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        # 缓存大小
        self.buf_size = buf_size
        # 缓存的内容
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    # 获取数据，存储在buffer中
    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if the buffer needs to be flushed to disk
        # 缓存size>阈值
        if len(self.buffer["data"]) >= self.buf_size:
            self.flush()

    # 将数据写入磁盘
    def flush(self):
        # write the buffers to disk then reset the buffer
        # 获取idx
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    # 存储数据中的label：表情的class
    def store_class_labels(self, class_labels):
        # create a dataset to store the actual class label names,
        # then store the class labels
        dt = h5py.special_dtype(vlen=str)
        label_set = self.db.create_dataset("label_names", (len(class_labels),), dtype=dt)
        label_set[:] = class_labels

    # 关闭写入和读取的流
    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()
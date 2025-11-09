# import the necessary packages
from tensorflow.keras.callbacks import Callback
import json
import os
import matplotlib.pyplot as plt
import numpy as np


class TrainingMonitor(Callback):
    def __init__(self, fig_path, json_path=None, start_at=0):
        # call the parent constructor
        super().__init__()

        # store the output paths for the figure and JSON files
        self.figPath = fig_path
        self.jsonPath = json_path

        # initialize the dictionary that will be used to store our training history
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None and os.path.exists(self.jsonPath):
            self.H = json.loads(open(self.jsonPath).read())

            # check to see if a starting epoch was supplied
            if start_at > 0:
                # loop over the entries in the history log and trim any entries that are past the starting epoch
                for k in self.H.keys():
                    self.H[k] = self.H[k][:start_at]

    def on_train_begin(self, logs=None):
        # initialize the history dictionary if it doesn't exist
        if self.H == {}:
            # Initialize with common metrics
            self.H["loss"] = []
            self.H["accuracy"] = []
            self.H["val_loss"] = []
            self.H["val_accuracy"] = []

    def on_epoch_end(self, epoch, logs=None):
        # loop over the metrics and add their values to the history
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        # check to see if the training history should be serialized to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

        # check to see if the training plot should be updated
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            n = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(n, self.H["loss"], label="train_loss")
            plt.plot(n, self.H["val_loss"], label="val_loss")
            plt.plot(n, self.H["accuracy"], label="train_acc")
            plt.plot(n, self.H["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # save the figure
            plt.savefig(self.figPath)
            plt.close()
